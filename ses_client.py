"""
Amazon SES sender.

Environment variables:
  SENDER_EMAIL   an SES-verified identity (in the same region as the Lambda)
  AWS_REGION     set automatically inside Lambda; for local runs, falls back to us-east-1

In SES sandbox mode, the recipient must ALSO be a verified identity.
"""
import logging
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

_client = None


def _ses():
    global _client
    if _client is None:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        _client = boto3.client("ses", region_name=region)
    return _client


def send_email(to_address, subject, body):
    """Send one plain-text email. Returns True on success, False on any failure."""
    sender = os.environ.get("SENDER_EMAIL")
    if not sender:
        logger.error("SENDER_EMAIL is not set; cannot send")
        return False
    try:
        _ses().send_email(
            Source=sender,
            Destination={"ToAddresses": [str(to_address).strip()]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        return True
    except (ClientError, BotoCoreError) as exc:
        logger.warning("SES send failed for %s (%s)", to_address, exc)
        return False
