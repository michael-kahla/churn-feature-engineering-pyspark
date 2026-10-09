# Churn Feature Engineering with PySpark on Databricks

Feature engineering pipeline that turns raw customer data from a food-delivery platform into **ML-ready features** for customer churn prediction. Built with **PySpark** and **Delta Lake** on **Databricks**, using Unity Catalog.

The goal is to prepare a clean, reusable **feature table**, the kind of asset a feature store serves to ML models for both training and inference.

## Pipeline overview

```
Raw customer data
   ↓ Delta table (customer_summary)
   ↓ Numerical + categorical features   → customer_features_v1
   ↓ Temporal features                  → customer_temporal_features
   ↓ Join on customer_id                → combined feature set
   ↓ Skewness check → log transform → Min-Max scaling → ordinal encoding
ML-ready features
```

## What the notebook does

### 1. Setup and storage
- Creates the Unity Catalog schema and writes the source data as a Delta table (`pipeline_catalog.tabular_pipeline.customer_summary`).

### 2. Numerical and categorical features
- **Binning** → `order_frequency_segment`, `rating_segment` (Low / Medium / High)
- **Binary flag** → `high_value_customer` (total spent above a business threshold)
- **One-hot encoding** → `payment_method` (few categories, no natural order)
- **Frequency encoding** → `preferred_cuisine` (many categories, one-hot would be too wide)
- **Boolean to 0/1** → `is_premium_flag`
- Saved as a versioned feature table: `customer_features_v1`

### 3. Temporal features
- **Tenure** → time between sign-up and last order (active lifetime)
- **Recency** → time since last order, segmented into Active / At Risk / Inactive
- **Engagement ratio** → orders relative to tenure (activity rate, not just totals)
- Signup month and quarter

Tenure and recency are complementary: two customers with the same tenure can have very different recency, and only recency shows that one of them is about to churn.

### 4. Combining feature sets
- Joins numerical, categorical and temporal features on `customer_id`
- Row count checked before and after the join to make sure no customer is lost (1,000 records)

### 5. Preprocessing for ML
- **Skewness check** with `pyspark.sql.functions.skewness` before transforming anything
- **Log transformation** on the highly skewed `engagement_ratio`
- **Min-Max scaling** on continuous features (`total_spent`, `total_orders`, `avg_order_value`, `engagement_ratio_log`) so no feature dominates because of its units
- **Ordinal encoding** on ordered segments (Low < Medium < High, Active < At Risk < Inactive)


## Key concepts covered
- Feature engineering vs. preprocessing (creating information vs. formatting it for the model)
- Choosing the right encoding: one-hot, ordinal or frequency
- Handling skewed distributions and scaling numerical features
- Feature table design: primary key, feature columns, timestamp, data types
- Feature versioning (`_v1`) for reproducibility
- Offline vs. online feature stores and training-serving skew
