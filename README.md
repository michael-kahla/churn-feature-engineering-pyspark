# Churn Feature Engineering with PySpark on Databricks

Feature engineering pipeline that turns raw customer data from a food-delivery platform into **ML-ready features** for customer churn prediction. It covers both **structured features** (spending, orders, tenure, recency) and **text features** from customer reviews (TF-IDF and LLM embeddings).

Built with **PySpark**, **Delta Lake** and **Unity Catalog** on **Databricks**.

## Pipeline overview

```
Raw customer data + reviews
   ↓ 01 · Structured features    → customer_features_v1, customer_temporal_features
   ↓ 02 · Text features (TF-IDF) → token counts, sentence length, TF-IDF arrays
   ↓ 03 · Embedding table design → customer_review_embeddings (schema + constraint)
   ↓ 04 · LLM embeddings         → generated with ai_query, stored with metadata
ML-ready feature tables
```

Notebooks
#	Notebook	Focus
- 01	**0_1_feature_engineering_for_ml_readiness _**--> Numerical, categorical and temporal features + preprocessing

   - **Binning**
   - **Binary flag**
   - **one-hot encoding**
   - **frequency encoding**
   - **temporal features** (tenure (sign-up to last order), recency (time since last order), engagement ratio (orders relative to tenure))
   - **preprocessing**(skewness check, log transfor, Min max scaling, ordinal encoding)
   - Feature sets joined on customer_id, with row counts checked before and after the join.

Tenure and recency are complementary: two customers with the same tenure can have very different recency, and only recency shows that one of them is about to churn.

- **02 · 02_text_features_tfidf** → Text cleaning, length features, TF-IDF
   - **Cleaning** → lowercase, trim, remove punctuation and special characters
   - **Length features** → `token_count`, `sentence_count`, `avg_sentence_length` (sentences detected on the original text, before punctuation is removed)
   - **Tokenization + stop-word removal** → `Tokenizer`, `StopWordsRemover`
   - **Term frequency** → `CountVectorizer` with `minDF=2` to drop words too rare to learn from (typos, one-off terms)
   - **TF-IDF** → `IDF` down-weights common words (`food` ≈ 0.45) and up-weights distinctive ones (`fresh` ≈ 2.83)
   - **Storage** → `vector_to_array` converts the sparse ML vector into a standard array that Delta can store  

- 03	**03_designing_embedding_feature_table**	Feature table schema for embeddings + data quality constraint
- **Delta table** `customer_review_embeddings`:
      - `customer_id` → key
      - `review_text` → source text, kept so embeddings can be regenerated
      - `embedding` → `ARRAY<DOUBLE>`
      - `embedding_model`, `embedding_dimension`, `generated_at` → model metadata
   - **Constraint** → `CHECK (embedding_dimension = 1024)` rejects vectors from a different model

   Unlike TF-IDF, embeddings come from an external model: their values only make sense relative to that model, so each row must record which model produced it.

- 04	**04_storing_embeddings_in_feature_table**	Real embedding generation and storage
   - **Sample** → 10 reviews, to limit calls to the model
   - **Embeddings** → `ai_query('databricks-gte-large-en', review_text)` returns a 1,024-dimension vector per review
   - **Metadata** → model name, dimension and timestamp added before writing
   - **Verification** → every stored vector has 1,024 values
   - **Constraint test** → inserting a 3-dimension embedding is rejected (`DELTA_VIOLATE_CONSTRAINT_WITH_VALUES`)
   