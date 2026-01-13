import streamlit as st
import pandas as pd

from services.predict import predict_churn_and_recommend

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Churn Prediction System",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction & Retention System")

st.markdown(
    """
    This application predicts whether a customer is likely to churn
    and provides **personalized retention recommendations**
    based on customer behavior and customer segmentation.
    """
)

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
st.sidebar.header("Customer Information")

with st.sidebar.expander("👤 Demographics", expanded=True):
    customer_age = st.slider("Customer Age", 18, 80, 40)
    dependent_count = st.slider("Dependent Count", 0, 5, 2)
    gender = st.selectbox("Gender", ["M", "F"])
    education = st.selectbox(
        "Education Level",
        ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"]
    )
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    income = st.selectbox(
        "Income Category",
        ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"]
    )

with st.sidebar.expander("💳 Account & Credit Info", expanded=True):
    credit_limit = st.number_input("Credit Limit", 1000, 50000, 12000)
    revolving_bal = st.number_input("Total Revolving Balance", 0, 20000, 1500)
    open_to_buy = st.number_input("Avg Open To Buy", 0, 50000, 10000)
    util_ratio = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.3)

with st.sidebar.expander("📊 Engagement & Transactions", expanded=True):
    months_on_book = st.slider("Months on Book", 1, 60, 36)
    total_relationship = st.slider("Total Relationship Count", 1, 6, 4)
    months_inactive = st.slider("Months Inactive (Last 12 Months)", 0, 6, 2)
    contacts_count = st.slider("Contacts Count (Last 12 Months)", 0, 6, 2)
    total_trans_amt = st.number_input("Total Transaction Amount", 0, 20000, 4000)
    total_trans_ct = st.slider("Total Transaction Count", 0, 150, 60)
    card = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])

# --------------------------------------------------
# Validation
# --------------------------------------------------
if open_to_buy > credit_limit:
    st.sidebar.warning("⚠ Avg Open To Buy should not exceed Credit Limit")

# --------------------------------------------------
# Input DataFrame
# --------------------------------------------------
input_df = pd.DataFrame([{
    "Customer_Age": customer_age,
    "Dependent_count": dependent_count,
    "Months_on_book": months_on_book,
    "Total_Relationship_Count": total_relationship,
    "Months_Inactive_12_mon": months_inactive,
    "Contacts_Count_12_mon": contacts_count,
    "Credit_Limit": credit_limit,
    "Total_Revolving_Bal": revolving_bal,
    "Avg_Open_To_Buy": open_to_buy,
    "Total_Trans_Amt": total_trans_amt,
    "Total_Trans_Ct": total_trans_ct,
    "Avg_Utilization_Ratio": util_ratio,
    "Gender": gender,
    "Education_Level": education,
    "Marital_Status": marital,
    "Income_Category": income,
    "Card_Category": card
}])

st.subheader("🔎 Model Input Summary")
st.dataframe(input_df, use_container_width=True)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Churn"):
    try:
        result = predict_churn_and_recommend(input_df)

        st.subheader("📊 Prediction Result")

        if result["Churn Prediction"] == "Churn":
            st.error(f"🚨 High Risk of Churn ({result['Churn Probability (%)']}%)")
        else:
            st.success(f"✅ Low Risk of Churn ({result['Churn Probability (%)']}%)")

        st.markdown(f"### 🧩 Customer Segment: **{result['Customer Segment']}**")

        st.subheader("🎯 Recommended Actions")
        for rec in result["Recommendation"]:
            st.markdown(f"🔥 **Priority {rec['priority']}**: {rec['action']}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
