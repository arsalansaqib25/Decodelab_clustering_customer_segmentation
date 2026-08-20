# 📊 Customer Segmentation Using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![K-Means](https://img.shields.io/badge/Algorithm-K--Means-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

**Decodelab_Clustering_Customer_Segmentation** is an unsupervised machine learning project developed to segment customers based on their purchasing behavior.

The project uses **K-Means Clustering** to identify groups of customers with similar characteristics, allowing businesses to better understand customer behavior and develop targeted marketing, retention, and sales strategies.

Instead of treating all customers the same, the project transforms transactional data into meaningful **customer-level behavioral features** and discovers natural customer segments using unsupervised learning.

---

## 🎯 Business Problem

Businesses often have thousands of customers with very different purchasing behaviors.

Some customers:

* Purchase frequently but spend relatively small amounts.
* Make fewer purchases but place very large orders.
* Purchase a wide variety of products.
* Have low engagement and limited purchasing activity.
* Have long-term relationships with the business.
* Frequently cancel orders.

Treating all these customers identically can lead to inefficient marketing and customer-retention strategies.

### Objective

The primary objective of this project is to:

> **Identify meaningful customer segments based on purchasing behavior and translate those segments into actionable business personas.**

---

## 🎯 Project Objectives

* Perform data cleaning and preprocessing.
* Convert transaction-level data into customer-level data.
* Perform exploratory data analysis.
* Engineer meaningful customer behavioral features.
* Identify redundant or unsuitable clustering features.
* Standardize numerical features.
* Determine the optimal number of clusters.
* Apply K-Means clustering.
* Evaluate clustering quality using **Inertia** and **Silhouette Score**.
* Profile the resulting customer segments.
* Create business-oriented customer personas.
* Provide actionable recommendations for each customer segment.

---

## 🧠 Machine Learning Approach

This project uses **unsupervised learning**, meaning the algorithm discovers patterns in the data without predefined target labels.

### Algorithm

**K-Means Clustering**

K-Means groups customers into clusters by minimizing the distance between customers and their assigned cluster centroids.

The general workflow is:

```text
Raw Transaction Data
        ↓
Data Cleaning
        ↓
Customer-Level Aggregation
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
Optimal K Selection
        ↓
K-Means Clustering
        ↓
Cluster Profiling
        ↓
Customer Personas
        ↓
Business Recommendations
```

---

# 📂 Dataset

The dataset contains transactional information that is aggregated at the customer level to construct behavioral features.

The original transaction data includes information related to:

* Customers
* Orders
* Products
* Quantities
* Unit prices
* Countries
* Cancellations
* Transaction dates



---

# 🛠️ Feature Engineering

Transaction-level records were transformed into customer-level behavioral features.

The final feature set includes:

### 💰 Monetary Features

* `Total_Spend`
* `Average_Order_Value`
* `Median_Order_Value`
* `Maximum_Order_Value`
* `Minimum_Order_Value`
* `Average_Unit_Price`
* `Maximum_Unit_Price`
* `Minimum_Unit_Price`
* `Spend_per_Transaction`

### 🛒 Quantity Features

* `Total_Quantity`
* `Average_Quantity`
* `Maximum_Quantity`
* `Minimum_Quantity`
* `Quantity_per_Transaction`

### 📦 Product Behavior

* `Unique_Products`
* `Unique_Descriptions`
* `Product_Diversity`

### 📈 Engagement Features

* `Total_Transactions`
* `Recency`
* `Customer_Lifetime`
* `Countries`

### ❌ Customer Behavior

* `Cancellation_Rate`

---

# ⚠️ CustomerID Exclusion

`CustomerID` is treated as an **identifier**, not a behavioral feature.

Therefore, it should not be used as an input feature for the final K-Means model.

It is retained separately so that the resulting cluster assignment can be linked back to individual customers.

This prevents the clustering algorithm from learning patterns based on arbitrary customer identification numbers.

---

# 📊 Exploratory Data Analysis

The exploratory analysis investigates:

* Missing values
* Duplicate transactions
* Numerical distributions
* Customer spending behavior
* Order value distributions
* Quantity distributions
* Transaction frequency
* Product diversity
* Customer recency
* Customer lifetime
* Cancellation behavior
* Feature correlations
* Potential outliers

### Key EDA Goals

The EDA was used to understand:

1. How customers differ in spending.
2. How frequently customers purchase.
3. How diverse their product choices are.
4. Which features are highly correlated.
5. Which features may require special treatment before clustering.

---

# 🔎 Feature Selection

Feature selection is particularly important for K-Means because the algorithm is distance-based.

Highly redundant features can cause certain behavioral dimensions to receive excessive influence.

The feature-selection process considers:

* Feature correlation
* Redundant information
* Business relevance
* Identifier columns
* Feature distributions
* Contribution to customer behavior representation

The objective is to retain features that represent different and meaningful dimensions of customer behavior.

---

# 📏 Feature Scaling

Numerical features are standardized before applying K-Means.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Standardization ensures that features with larger numerical ranges do not dominate the distance calculations.

---

# 🔢 Selecting the Optimal Number of Clusters

Multiple values of `K` were evaluated to determine an appropriate number of customer segments.

Two primary evaluation methods were used:

### 1. Elbow Method

The **inertia** measures the within-cluster sum of squared distances.

Lower inertia indicates more compact clusters.

However, increasing `K` will naturally reduce inertia, so the objective is to identify an appropriate point where additional clusters provide diminishing improvement.

### 2. Silhouette Score

The silhouette score measures how well each customer fits within its assigned cluster compared with neighboring clusters.

A higher score generally indicates better-separated and more cohesive clusters.

---

# 🤖 Final Model

After evaluating different values of `K`, the final model uses:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)
```

### Final Configuration

| Parameter          | Value                      |
| ------------------ | -------------------------- |
| Algorithm          | K-Means                    |
| Number of Clusters | **3**                      |
| Initialization     | `k-means++`                |
| Scaling            | `StandardScaler`           |
| Random State       | `42`                       |
| Evaluation         | Inertia + Silhouette Score |

---

# 👥 Customer Segmentation Results

The final K-Means model identified **three distinct customer segments**.

## 🟢 Cluster 0: Frequent & Diverse Buyers

### Customer Profile

These customers demonstrate:

* High transaction frequency
* High product diversity
* High number of unique products
* Long customer lifetime
* Recent activity
* Above-average total spending
* Above-average total quantity
* Relatively small individual transactions
* Higher cancellation activity

### Business Persona

> **Frequent & Diverse Buyers**

These customers generate value primarily through **frequent interactions and broad product engagement** rather than very large individual transactions.

### Recommended Strategy

* Loyalty and rewards programs
* Personalized product recommendations
* Cross-selling
* Bundle offers
* Encourage larger basket sizes
* Reduce cancellation behavior

---

## 🔵 Cluster 1: Low-Engagement Customers

### Customer Profile

These customers show:

* Very low total spending
* Low transaction frequency
* Low quantity
* Low order values
* Low product diversity
* Few unique products
* Shorter customer lifetime
* Older recency
* Low spend per transaction
* Very low cancellation rate

### Business Persona

> **Low-Engagement Customers**

These customers represent the lowest-engagement segment and may require targeted campaigns to increase purchasing activity.

### Recommended Strategy

* Re-engagement campaigns
* Personalized discounts
* Limited-time offers
* Product recommendations
* First/next purchase incentives
* Customer retention campaigns

---

## 🟣 Cluster 2: High-Value Bulk Buyers

### Customer Profile

These customers demonstrate:

* High total spending
* Very high average order value
* Very high median order value
* Very high minimum and maximum order values
* High total quantity
* Very high quantity per transaction
* Very high spend per transaction
* High unit prices
* Relatively recent activity
* Low cancellation rate
* Low product diversity

### Business Persona

> **High-Value Bulk Buyers**

These customers generate substantial value through **large and expensive transactions** rather than frequent purchases across a wide product range.

### Recommended Strategy

* VIP customer treatment
* Bulk-order discounts
* Volume-based pricing
* Premium product recommendations
* Dedicated account management
* Cross-selling complementary products
* Priority customer support

---

# 📌 Final Customer Persona Summary

| Cluster       | Persona                      | Primary Behavior                         | Business Focus            |
| ------------- | ---------------------------- | ---------------------------------------- | ------------------------- |
| **Cluster 0** | 🟢 Frequent & Diverse Buyers | High frequency + broad product diversity | Loyalty & cross-selling   |
| **Cluster 1** | 🔵 Low-Engagement Customers  | Low activity + low spending              | Re-engagement & retention |
| **Cluster 2** | 🟣 High-Value Bulk Buyers    | Large transactions + high spending       | VIP & bulk-sales strategy |

---

# 💼 Business Insights

The clustering results demonstrate that customer value is not determined by spending alone.

### Cluster 0

Customer value is driven by:

**Frequency + Product Diversity**

### Cluster 1

Customer value is limited by:

**Low Engagement + Low Spending**

### Cluster 2

Customer value is driven by:

**Large Transactions + High Spend per Transaction**

This distinction allows businesses to design different strategies for different customer behaviors instead of applying a single marketing strategy to the entire customer base.

---

# 📈 Visualizations

The project includes visualizations for:

* Missing-value analysis
* Feature distributions
* Correlation heatmap
* Outlier analysis
* Elbow curve
* Silhouette scores
* Cluster distribution
* PCA-based cluster visualization
* Customer persona comparison

Example project visualizations can be stored in:

```text
reports/
└── figures/
```

---

# 📁 Project Structure

```text
Decodelab_Clustering_Customer_Segmentation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_Model_training.ipynb
│
├── model/
│   └── .gitkeep
│
├── reports/
   ├── figures/
   └── customer_segmentation_report.md

```

---

# 🧰 Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**
* **K-Means Clustering**
* **StandardScaler**
* **PCA**
* **Git & GitHub**

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/arsalansaqib25/Decodelab_clustering_customer_segmentation.git
```

Navigate to the project directory:

```bash
cd Decodelab_Clustering_Customer_Segmentation
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

# ▶️ Usage

Follow the notebooks in order:

```text
01_data_understanding
        ↓
02_preprocessing
        ↓
03_EDA
        ↓
04_Model_training
```

This order ensures that the complete data science workflow can be reproduced from data understanding through business interpretation.

---

# 📊 Key Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Unsupervised Learning
* Customer Segmentation
* K-Means Clustering
* Feature Engineering
* Feature Selection
* Feature Scaling
* Distance-Based Algorithms
* Elbow Method
* Silhouette Analysis
* PCA Visualization
* Cluster Profiling
* Business Persona Development
* Data-Driven Business Recommendations

---

# 🚀 Future Improvements

Potential extensions include:

* Compare K-Means with **DBSCAN** and **Hierarchical Clustering**
* Apply dimensionality reduction before clustering
* Experiment with robust scaling
* Perform automated feature selection
* Build an interactive customer segmentation dashboard
* Track customer segments over time
* Develop customer lifetime value analysis
* Integrate RFM-based segmentation
* Build a deployment-ready segmentation API
* Monitor cluster stability on new customer data

---

# 👨‍💻 Author

**Arsalan Saqib**

Bachelor of Science in Computer Science

### Data Science Intern at DecodeLabs

This project was developed as part of practical data science and machine learning work at **DecodeLabs**, focusing on customer analytics and unsupervised machine learning.

---

# 📜 License

This project is intended for educational and portfolio purposes.

If a specific license is required for the dataset, follow the original dataset provider's licensing and usage terms.

---

## ⭐ Acknowledgment

Special thanks to **DecodeLabs** for providing the opportunity to work on practical data science and machine learning projects.

If you found this project useful, consider giving the repository a ⭐.
