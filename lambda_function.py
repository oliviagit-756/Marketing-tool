"""
Lambda handler.

Flow: dataset CSV in S3 (with First_Name + Email columns) -> RFM features -> segment (saved JSON model)
      -> Groq templates (one per segment) -> fill per customer -> SES send -> log CSV to S3.

Environment variables:
  MODEL_KEY      default model/segmentation_model.json
  BUCKET         bucket for manual test events (S3 events carry their own bucket)
  SENDER_EMAIL   SES-verified sender
  GROQ_API_KEY, GROQ_MODEL   for message generation
  DRY_RUN        "true" (default) = generate and log only, send nothing. Set "false" to send.
  PER_SEGMENT    customers emailed per segment per run, default 1
  MAX_EMAILS     hard safety cap per run, default 10

Manual test event: {"transactions_key": "uploads/data_with_email.csv"}

Suggested Lambda settings: memory 2048 MB, timeout 300 s (the dataset is ~58 MB).
"""
import io
import json
import logging
import os
from datetime import datetime, timezone
from urllib.parse import unquote_plus
import boto3
import pandas as pd

from groq_client import generate_templates, render_message
from model import assign_segments, build_customer_features
from ses_client import send_email

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = os.environ.get("BUCKET")
MODEL_KEY = os.environ.get("MODEL_KEY", "model/segmentation_model.json")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"
PER_SEGMENT = int(os.environ.get("PER_SEGMENT", "1"))
MAX_EMAILS = int(os.environ.get("MAX_EMAILS", "10"))

NEEDED_COLUMNS = ["InvoiceNo", "Description", "Quantity", "InvoiceDate",
                  "UnitPrice", "CustomerID", "First_Name", "Email"]

_bundle = None  # cached across warm invocations


def _clean_product(product):
    product = str(product).strip().title()
    return product[:45] + "..." if len(product) > 45 else product


def _extract_contacts(transactions):
    """One row per customer with a usable email: CustomerID, First_Name, Email."""
    contacts = transactions[["CustomerID", "First_Name", "Email"]].dropna()
    contacts = contacts[contacts["Email"].astype(str).str.strip() != ""]
    contacts = contacts.assign(CustomerID=contacts["CustomerID"].astype(float).astype(int).astype(str))
    return contacts.drop_duplicates("CustomerID")


def process(transactions, bundle, templates, sender=send_email,
            dry_run=True, per_segment=1, max_emails=10):
    """Pure logic (no AWS calls except `sender`), so it can be tested locally."""
    features = assign_segments(build_customer_features(transactions), bundle)
    features["Top_Product"] = features["Top_Product"].map(_clean_product)

    contacts = _extract_contacts(transactions)
    merged = features.merge(contacts, on="CustomerID", how="inner")
    targets = merged.groupby("Segment", group_keys=False).head(per_segment).head(max_emails)

    rows = []
    for r in targets.itertuples():
        subject, body = render_message(templates[r.Segment], str(r.First_Name), r.Top_Product)
        status = "dry_run"
        if not dry_run:
            status = "sent" if sender(r.Email, subject, body) else "failed"
        rows.append({
            "CustomerID": r.CustomerID, "Segment": r.Segment, "Email": r.Email,
            "Template_Source": templates[r.Segment]["source"],
            "Subject": subject, "Body": body, "Status": status,
        })
    return features, pd.DataFrame(rows)


def _resolve_input(event):
    if "Records" in event:  # S3 trigger
        rec = event["Records"][0]["s3"]
        return rec["bucket"]["name"], unquote_plus(rec["object"]["key"])
    return event.get("bucket", BUCKET), event["transactions_key"]  # manual test


def _load_bundle(s3, bucket):
    global _bundle
    if _bundle is None:
        obj = s3.get_object(Bucket=bucket, Key=MODEL_KEY)
        _bundle = json.loads(obj["Body"].read())
    return _bundle


def _read_csv(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), encoding="latin1", usecols=NEEDED_COLUMNS)


def lambda_handler(event, context):
    s3 = boto3.client("s3")
    bucket, tx_key = _resolve_input(event)

    transactions = _read_csv(s3, bucket, tx_key)
    bundle = _load_bundle(s3, bucket)
    templates = generate_templates()  # one Groq call per segment

    features, log = process(transactions, bundle, templates, dry_run=DRY_RUN,
                            per_segment=PER_SEGMENT, max_emails=MAX_EMAILS)

    log_key = f"logs/run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.csv"
    s3.put_object(Bucket=bucket, Key=log_key, Body=log.to_csv(index=False).encode("utf-8"))

    summary = {
        "customers_segmented": int(len(features)),
        "segment_counts": features["Segment"].value_counts().to_dict(),
        "messages_generated": int(len(log)),
        "sent": int((log["Status"] == "sent").sum()) if len(log) else 0,
        "failed": int((log["Status"] == "failed").sum()) if len(log) else 0,
        "dry_run": DRY_RUN,
        "log_key": log_key,
    }
    logger.info(summary)
    return summary
