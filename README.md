

# 💳 Credit Card Fraud Detection AI using DBSCAN (Unsupervised ML)

---

## **Project Idea & Motivation**

Credit card fraud is a growing problem in the digital payment ecosystem. Detecting fraudulent transactions is challenging because:

* Fraud is **rare compared to normal transactions**.
* Fraud patterns are **irregular and evolving**, making rule-based systems ineffective.
* Many datasets are **unlabeled**, limiting supervised learning approaches.

**Solution:** Use **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**, an **unsupervised ML algorithm**:

* Detects **dense clusters** of normal transactions.
* Labels **isolated points** as **outliers**, potentially fraudulent transactions.
* Works without requiring prior labels.

This project creates an **interactive Streamlit application** that allows users to:

1. Upload a real dataset or generate a random sample dataset.
2. Preprocess and scale transaction data automatically.
3. Apply DBSCAN to detect clusters and outliers.
4. Visualize clusters and outliers in **interactive PCA plots**.
5. Download outlier transactions as CSV for reporting or analysis.

---

## **Project Flow & Steps**

### **Step 1: Data Input**

* Users can either:

  * **Upload a CSV** dataset with numeric transaction features.
  * **Generate a random dataset** for demonstration (950 normal + 50 fraudulent transactions).
* Uses **Streamlit file uploader** and buttons.
* Displays a **preview of the dataset**.

### **Step 2: Preprocessing**

* Drop the `Class` column if present (used only for evaluation).
* Scale features using **StandardScaler**:

  * Ensures all numeric features have zero mean and unit variance.
  * Handles feature-name-safe scaling to prevent mismatches.

### **Step 3: DBSCAN Parameter Adjustment**

* User-defined sliders for:

  * **Epsilon (eps)**: Maximum distance to consider points as neighbors.
  * **Minimum Samples (min_samples)**: Minimum points to form a cluster.
* Real-time parameter adjustment allows **dynamic experimentation**.

### **Step 4: Apply DBSCAN**

* Fit the **DBSCAN algorithm** on scaled features.
* Labels points as:

  * `0, 1, 2, …` → clusters of normal transactions
  * `-1` → outliers (potential fraud)

### **Step 5: PCA Visualization**

* Apply **Principal Component Analysis (PCA)** to reduce features to 2D.
* Visualize clusters and outliers in an **interactive scatter plot** using Plotly.
* Outliers are **highlighted distinctly** for easy identification.

### **Step 6: Results Display**

* Show **key metrics**:

  * Number of clusters detected.
  * Number of outliers detected (potential fraud).
* Tabs:

  1. **Visualization:** PCA scatter plot of clusters & outliers.
  2. **Outlier Transactions:** Table with outlier data and **download option**.
  3. **Ground Truth Comparison:** If `Class` exists, shows **recall of detected frauds**.

### **Step 7: Optional Enhancements**

* Save the **scaler** using Joblib for reuse.
* Refitted automatically if dataset columns mismatch.
* Dark, modern UI with bold fonts and gradients for professional appearance.

---

## **Key Features**

* **Unsupervised Learning:** No labeled data required.
* **Random Dataset Generator:** Demo mode for quick experiments.
* **Interactive DBSCAN Parameters:** Epsilon & min_samples sliders.
* **Automatic Preprocessing:** Feature scaling and column handling.
* **PCA 2D Visualization:** Interactive cluster/outlier plots.
* **Outlier Detection & Download:** Export potential frauds to CSV.
* **Ground Truth Comparison:** Optional evaluation if dataset has `Class` column.
* **Professional UI:** Dark theme, modern fonts, and clean layout.

---

## **Technology Stack**

| **Technology**                  | **Purpose**                              |
| ------------------------------- | ---------------------------------------- |
| Python 3.9+                     | Core programming language                |
| Streamlit                       | Interactive web app interface            |
| Pandas, NumPy                   | Data manipulation and preprocessing      |
| Scikit-learn                    | StandardScaler, DBSCAN, PCA              |
| Plotly                          | Interactive visualizations               |
| Joblib                          | Save/load scaler for feature consistency |
| Matplotlib / Seaborn (optional) | Additional plots if needed               |
| CSS                             | Modern UI customization                  |

---

## **Installation Instructions**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/dbscan-fraud-detection.git
cd dbscan-fraud-detection
```

2. Create virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## **Usage Instructions**

1. Upload your CSV dataset **or generate a random sample dataset**.
2. Adjust **DBSCAN parameters** (eps, min_samples) via sidebar sliders.
3. Click **Run Analysis** → DBSCAN clusters data and identifies outliers.
4. Explore tabs:

   * **Visualization:** PCA 2D interactive plot.
   * **Outliers:** Table of potential frauds & CSV download.
   * **Ground Truth:** Compare predicted outliers with `Class` if available.

---

## **DBSCAN Overview**

* Density-Based Spatial Clustering of Applications with Noise.
* Groups points into clusters based on density.
* **Outliers (-1)** represent sparse points isolated from clusters.
* Advantages for fraud detection:

  * Works with unlabeled data.
  * Detects anomalies without prior knowledge of fraud patterns.
  * Handles irregular cluster shapes.

---

## **Project Structure**

```
dbscan-fraud-detection/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── scaler.pkl             # Saved StandardScaler (optional)
├── dbscan_model.pkl       # Saved DBSCAN model (optional)
├── sample_data.csv        # Optional generated dataset
└── README.md              # Project documentation
```

---

## **Highlights for CV / Portfolio**

* Implemented **unsupervised ML pipeline** for fraud detection.
* Built **interactive, professional Streamlit dashboard**.
* Applied **DBSCAN clustering**, **PCA visualization**, and **outlier detection**.
* Integrated **automatic preprocessing and feature scaling** with Joblib.
* Added **dynamic visualization, download options, and ground truth evaluation**.

---

## **Future Enhancements**

* **Parameter auto-tuning** using k-distance plots.
* **Confusion matrix & metrics** for labeled datasets.
* **Real-time alerts** for detected fraud transactions.
* **Integration with real-time transaction APIs**.
* **Enhanced dashboard analytics** for CV or client presentation.

---

## **References**

1. [DBSCAN – Scikit-learn Documentation](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
2. [Streamlit Documentation](https://docs.streamlit.io/)
3. [Plotly for Python](https://plotly.com/python/)
4. [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## **License**

 Developed by **Siddhant Patel**

---


