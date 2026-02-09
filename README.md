# Superstore Intelligence Engine v2.0

This repository contains the source code for an enterprise-grade MLOps platform designed to provide customer intelligence and operational insights for e-commerce.

Moving beyond simple notebooks, this project demonstrates a **modular, production-ready pipeline** that handles data engineering, generative synthetic augmentation, parallel model training, and explainable AI deployment.

---

## 🚀 Live Demo

**[https://superstore-intelligence-v2.streamlit.app/](https://superstore-intelligence-v2.streamlit.app/)**

---

## 🏛️ Project Architecture

This platform is architected as a sequential MLOps pipeline, emphasizing reproducibility, schema enforcement, and the prevention of training-serving skew.

### **1. Data Engineering & Augmentation (`src/`)**
* **`01_ingest_and_clean.py`**: The ETL pipeline. Ingests raw CSVs, enforces strict Parquet schema, removes statistical outliers (IQR method), and fixes domain logic errors.
    * **Output:** `data/processed/cleaned_superstore_data.parquet`
* **`02_auto_eda_report.py`**: Automated health checks. Generates a comprehensive visual report to validate data distribution before training begins.
    * **Output:** `reports/comprehensive_eda_report/*.png`
* **`03_augment_data_gen.py`**: The Generative AI layer. Trains a **CTGAN** to learn customer distributions and generates 30,000+ synthetic transactions.
    * **Output:** `data/processed/synthetic_base_transactions.parquet`
    * **Model Artifact:** `models/ctgan_synthesizer.pkl` (Saved generator for future sampling)

### **2. The Model Training Farm (`src/`)**
* **`04a_train_profit_engine.py`**: Trains the "Deal Desk" logic using a two-stage approach (Risk + Magnitude) on a hybrid (Real + Synthetic) dataset.
    * **Output Artifacts:**
        * `models/final_v3_profitability_classifier.joblib` (The Gatekeeper)
        * `models/final_v3_profit_forecaster.joblib` (The Estimator)
* **`04b_train_segmentation.py`**: Trains a **Keras Autoencoder** to compress RFM data into a latent space for non-linear clustering.
    * **Output Artifacts:**
        * `models/customer_encoder_model.keras` (Neural Network)
        * `models/rfm_scaler.joblib` (Required for normalizing input data)
* **`04c_train_recommender.py`**: Trains a "Predictive Market Basket" model to predict product compatibility based on item features.
    * **Output Artifacts:**
        * `models/final_predictive_mba_classifier.joblib`

### **3. Audit & Explainability**
* **`05_generate_explainability.py`**: Generates **SHAP** plots to provide "White Box" transparency for the profit models.
    * **Output:** `reports/shap_summary_beeswarm.png`, `reports/shap_summary_bar.png`

### **4. Shared Logic (`common/`)**
* **`features.py`**: A centralized Feature Store module. Contains the `create_master_feature_set()` logic used by both Training and Inference pipelines to mathematically guarantee **Zero Training-Serving Skew**.

### **5. Deployment (`app.py`)**
* A multi-page **Streamlit** dashboard serving real-time inferences.
* **AI Strategy Co-Pilot**: Integrates **Gemini Pro** to read model outputs and draft strategic business memos for executives.

---

## 🔧 Key Technologies

* **Orchestration:** Modular Python Scripts (Airflow-ready)
* **Data Processing:** Pandas, Parquet, NumPy
* **Machine Learning:** Scikit-learn, XGBoost, TensorFlow/Keras
* **Generative AI:** SDV (CTGAN), Google Gemini
* **Interpretability:** SHAP
* **App Framework:** Streamlit

---

## ⚙️ Setup and Execution

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/superstore-intelligence.git](https://github.com/your-username/superstore-intelligence.git)
    cd superstore-intelligence
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Full MLOps Pipeline:**
    *The pipeline is designed to be executed sequentially.*

    ```bash
    # 1. ETL & Cleaning (Produces cleaned_superstore_data.parquet)
    python src/01_ingest_and_clean.py

    # 2. (Optional) Run Automated Health Checks
    python src/02_auto_eda_report.py

    # 3. Generate Synthetic Data (Produces synthetic_base_transactions.parquet)
    python src/03_augment_data_gen.py

    # 4. Train Models (Parallelizable)
    python src/04a_train_profit_engine.py
    python src/04b_train_segmentation.py
    python src/04c_train_recommender.py

    # 5. Generate Audit Reports (SHAP)
    python src/05_generate_explainability.py
    ```

4.  **Run the Streamlit Application:**
    *Ensure you have your Google AI API key set in `.streamlit/secrets.toml`.*
    ```bash
    streamlit run app.py
    ```

---

## ❓ Architectural FAQ

For a deep dive into the reasoning behind using **Parquet over CSV**, **Autoencoders vs K-Means**, and **Predictive MBA vs Apriori**, please see the detailed **[Architectural FAQ](./docs/FAQ.md)**.