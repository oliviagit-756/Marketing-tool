"""
Customer segmentation (Phase 1).
- build_customer_features(): raw transactions -> one row per customer (RFM + top product)
- train_segmentation():      fit scaler + KMeans, name clusters by RFM rank, return a JSON-friendly bundle
- assign_segments():         apply a saved bundle to customer features (numpy only, safe for Lambda)

scikit-learn is imported ONLY inside train_segmentation(), so the Lambda package
does not need it.
"""
import json

import numpy as np
import pandas as pd

RFM_COLS = ["Recency", "Frequency", "Monetary"]

# Ordered best -> worst. Cluster ranking decides which cluster gets which entry.
# "goal" and "tone" feed the LLM prompt; "fallback" is used if the LLM call fails.
SEGMENTS = [
    {
        "name": "Champions / VIP Customers",
        "goal": "Reward and retain high-value customers",
        "tone": "warm, exclusive, appreciative",
        "fallback": "Thank you for being one of our top customers! Enjoy an exclusive VIP reward on {product}.",
    },
    {
        "name": "Loyal Customers",
        "goal": "Increase repeat purchases and basket size",
        "tone": "friendly, grateful",
        "fallback": "We appreciate your loyalty! Get a special offer on {product}.",
    },
    {
        "name": "Recent / Occasional Buyers",
        "goal": "Turn recent low-frequency buyers into repeat customers",
        "tone": "friendly, encouraging",
        "fallback": "Enjoyed {product}? Here is something special for your next order.",
    },
    {
        "name": "Inactive / Lost Customers",
        "goal": "Win back inactive customers",
        "tone": "empathetic, 'we miss you'",
        "fallback": "We miss you! Come back and enjoy a special discount on {product}.",
    },
]


def build_customer_features(df, reference_date=None):
    """Clean raw transactions and return one row per customer: RFM + Top_Product."""
    df = df.copy()

    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["Total"] = df["Quantity"] * df["UnitPrice"]

    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Total", "sum"),
        )
        .reset_index()
    )

    top_product = (
        df.groupby(["CustomerID", "Description"])["Quantity"].sum().reset_index()
        .sort_values(["CustomerID", "Quantity"], ascending=[True, False])
        .drop_duplicates("CustomerID")[["CustomerID", "Description"]]
        .rename(columns={"Description": "Top_Product"})
    )

    features = rfm.merge(top_product, on="CustomerID", how="left")
    features["Top_Product"] = features["Top_Product"].fillna("your favorite products")
    return features


def train_segmentation(features, n_clusters=len(SEGMENTS), random_state=42):
    """Fit scaler + KMeans and auto-name clusters by RFM quality. Returns a JSON-friendly bundle."""
    from sklearn.cluster import KMeans  # training only, not needed in Lambda
    from sklearn.preprocessing import StandardScaler

    X = np.log1p(features[RFM_COLS])  # log tames the skew in Frequency/Monetary
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(X_scaled)

    # Score each cluster centre: low recency is good, high frequency/monetary is good.
    centers = kmeans.cluster_centers_
    score = -centers[:, 0] + centers[:, 1] + centers[:, 2]
    best_to_worst = np.argsort(-score)
    cluster_to_segment = {int(c): SEGMENTS[rank]["name"] for rank, c in enumerate(best_to_worst)}

    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "centers": centers.tolist(),
        "cluster_to_segment": cluster_to_segment,
    }


def assign_segments(features, bundle):
    """Apply a trained bundle (nearest centroid = what KMeans.predict does)."""
    features = features.copy()
    X = np.log1p(features[RFM_COLS].to_numpy(dtype=float))
    X = (X - np.array(bundle["mean"])) / np.array(bundle["scale"])
    centers = np.array(bundle["centers"])
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    features["Cluster"] = dists.argmin(axis=1)
    mapping = {int(k): v for k, v in bundle["cluster_to_segment"].items()}
    features["Segment"] = features["Cluster"].map(mapping)
    return features


def add_fallback_campaign(features):
    """Template-based messages (used by the Streamlit app; also the LLM fallback)."""
    by_name = {s["name"]: s for s in SEGMENTS}
    features = features.copy()
    features["Campaign_Goal"] = features["Segment"].map(lambda s: by_name[s]["goal"])
    features["Campaign_Message"] = [
        by_name[seg]["fallback"].format(product=prod)
        for seg, prod in zip(features["Segment"], features["Top_Product"])
    ]
    return features


def run_customer_segmentation(df, model_path=None):
    """Raw transactions -> segmented customers with fallback messages."""
    features = build_customer_features(df)
    bundle = train_segmentation(features)
    if model_path:
        with open(model_path, "w") as f:
            json.dump(bundle, f, indent=2)
    features = assign_segments(features, bundle)
    return add_fallback_campaign(features)


if __name__ == "__main__":
    # Quick self-test with synthetic transactions
    rng = np.random.default_rng(0)
    n = 6000
    demo = pd.DataFrame({
        "InvoiceNo": rng.integers(10000, 12500, n).astype(str),
        "CustomerID": rng.integers(12000, 12600, n).astype(float),
        "Description": rng.choice(["MUG", "LAMP", "CANDLE", "TEAPOT", "BAG"], n),
        "Quantity": rng.integers(1, 12, n),
        "UnitPrice": rng.uniform(1, 30, n).round(2),
        "InvoiceDate": pd.Timestamp("2011-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    })
    out = run_customer_segmentation(demo, model_path="segmentation_model.json")
    print(out["Segment"].value_counts())
    print(out.groupby("Segment")[RFM_COLS].mean().round(1))
