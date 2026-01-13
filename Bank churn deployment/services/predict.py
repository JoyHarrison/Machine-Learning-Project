import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Model paths (SAFE & PORTABLE)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

churn_pipeline = joblib.load(MODELS_DIR / "churn_pipeline.pkl")
kmeans = joblib.load(MODELS_DIR / "customer_segmentation.pkl")
cluster_scaler = joblib.load(MODELS_DIR / "cluster_scaler.pkl")

# --------------------------------------------------
# Config
# --------------------------------------------------
CHURN_THRESHOLD = 0.5

segment_names = {
    0: "Credit-Constrained Active Customers",
    1: "Low-Engagement High-Risk Customers",
    2: "Loyal Conservative Customers",
    3: "Premium High-Value Customers"
}

# --------------------------------------------------
# Recommendation engine
# --------------------------------------------------
def get_recommendation(segment_name, churn_label):
    actions = {
        "Credit-Constrained Active Customers": [
            "Increase credit limit based on recent spending behavior.",
            "Offer flexible repayment or EMI conversion options.",
            "Provide transaction-based reward incentives."
        ],
        "Low-Engagement High-Risk Customers": [
            "Provide cashback offers to encourage card usage.",
            "Send personalized re-engagement emails and notifications.",
            "Offer fee waivers or limited-time discounts."
        ],
        "Loyal Conservative Customers": [
            "Offer loyalty rewards and anniversary benefits.",
            "Introduce low-risk financial products.",
            "Provide personalized retention incentives."
        ],
        "Premium High-Value Customers": [
            "Assign a dedicated relationship manager.",
            "Provide exclusive premium benefits and concierge services.",
            "Offer customized high-value retention packages."
        ]
    }

    base_actions = actions.get(segment_name, ["Maintain customer relationship."])

    if churn_label == "Churn":
        base_actions.insert(0, "Immediate retention intervention required.")

    return [
        {"priority": i + 1, "action": action}
        for i, action in enumerate(base_actions)
    ]

# --------------------------------------------------
# Main function
# --------------------------------------------------
def predict_churn_and_recommend(customer_df: pd.DataFrame):
    if customer_df.shape[0] != 1:
        raise ValueError("Input DataFrame must contain exactly one row")

    # ---- Churn Prediction ----
    churn_prob = churn_pipeline.predict_proba(customer_df)[0][1]
    churn_label = "Churn" if churn_prob >= CHURN_THRESHOLD else "No Churn"

    # ---- Customer Segmentation ----
    cluster_features = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Avg_Utilization_Ratio"
    ]

    cluster_input = customer_df[cluster_features]
    cluster_scaled = cluster_scaler.transform(cluster_input)
    segment_id = kmeans.predict(cluster_scaled)[0]
    segment_name = segment_names.get(segment_id, "Unknown Segment")

    # ---- Recommendations ----
    recommendations = get_recommendation(segment_name, churn_label)

    return {
        "Churn Prediction": churn_label,
        "Churn Probability (%)": round(churn_prob * 100, 2),
        "Customer Segment": segment_name,
        "Recommendation": recommendations
    }
