"""Local DRY RUN: same logic as the Lambda, nothing is sent. Needs .env with GROQ_API_KEY and GROQ_MODEL."""
import json

import pandas as pd

from groq_client import generate_templates
from lambda_function import NEEDED_COLUMNS, process

transactions = pd.read_csv("data_with_email.csv", encoding="latin1", usecols=NEEDED_COLUMNS)
with open("segmentation_model.json") as f:
    bundle = json.load(f)

templates = generate_templates()
features, log = process(transactions, bundle, templates, dry_run=True, per_segment=1, max_emails=10)

print(features["Segment"].value_counts())
for r in log.itertuples():
    print(f"\n=== {r.Segment} [{r.Template_Source}] -> {r.Email} ===")
    print("Subject:", r.Subject)
    print(r.Body)

log.to_csv("local_dry_run_log.csv", index=False)
print("\nSaved local_dry_run_log.csv")
