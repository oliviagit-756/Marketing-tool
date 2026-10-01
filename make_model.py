"""Train K-Means on the full dataset and save segmentation_model.json (run locally, once)."""
import json

import pandas as pd

from model import assign_segments, build_customer_features, train_segmentation

df = pd.read_csv("data_with_email.csv", encoding="latin1")
features = build_customer_features(df)
bundle = train_segmentation(features)

with open("segmentation_model.json", "w") as f:
    json.dump(bundle, f, indent=2)

segmented = assign_segments(features, bundle)
print("Customers:", len(segmented))
print("Cluster -> segment:", bundle["cluster_to_segment"])
print(segmented["Segment"].value_counts())
print(segmented.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1))
print("\nSaved segmentation_model.json")
