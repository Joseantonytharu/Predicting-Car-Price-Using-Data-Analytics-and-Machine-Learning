# 🚗 Predicting Car Price Using Data Analytics and Machine Learning

A machine learning project that predicts automobile prices using regression and classification models, built on the classic UCI Automobile dataset.

---

## 📌 Project Overview

This project explores the relationship between vehicle characteristics and market price. It covers the full data science workflow — from exploratory analysis and preprocessing through to model training, evaluation, and comparison.

**Key goals:**
- Identify which vehicle features most influence price
- Build and compare regression models to predict car prices
- Classify vehicles into price categories using logistic regression

---

## 📂 Dataset

The dataset used is the [UCI Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/automobile), loaded from a local CSV file (`auto.csv`).

It contains **26 features** per vehicle, including:

| Type | Features |
|---|---|
| Physical | `wheel-base`, `length`, `width`, `height`, `curb-weight` |
| Engine | `engine-type`, `engine-size`, `horsepower`, `num-of-cylinders` |
| Performance | `city-mpg`, `highway-mpg`, `compression-ratio`, `peak-rpm` |
| Identity | `make`, `fuel-type`, `body-style`, `drive-wheels` |
| Target | `price` |

> **Note:** The raw dataset has no column headers — these are assigned manually in the notebook.

---

## 🔧 Technologies Used

- **Python 3**
- **pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — data visualization
- **scikit-learn** — machine learning models and evaluation

---

## 🗂️ Project Structure

```
├── Predicting_Car_Price_Using_Data_Analytics_and_Machine_Learning.ipynb
├── auto.csv                  # Dataset (required)
└── README.md
```

---

## 🔍 Workflow

### 1. Exploratory Data Analysis (EDA)
- Histograms of `price` and `horsepower`
- Boxplot of `price` by `body-style`
- Correlation heatmap across all numeric features

### 2. Data Preprocessing
- Replaced `"?"` placeholder values with `NaN`
- Converted relevant columns to numeric types
- Imputed missing values using **mean** (numeric) and **mode** (categorical)
- Detected and capped outliers using the **IQR / Winsorization** method

### 3. Feature Engineering
- Converted `highway-mpg` and `city-mpg` to liters per 100 km (`L/100km`)
- Binned `horsepower` into Low / Medium / High categories
- Normalized `engine-size` to a 0–1 scale

### 4. Regression Models
Models were trained and compared using two train/test splits:

| Split | Linear Regression | Random Forest |
|---|---|---|
| 80/20 | ✅ | ✅ |
| 70/30 | ✅ | ✅ |

**Evaluation metrics:** R² Score, MAE, MSE

> **Best performer:** Random Forest Regressor (80/20 split)

### 5. Classification Model
- Target variable `price` was bucketed into three classes: **Low**, **Medium**, **High** using quantile binning
- A **Logistic Regression** classifier was trained on the same feature set
- Evaluated using **Accuracy Score** and a **Confusion Matrix heatmap**

---

## 📊 Key Findings

- **Engine size** and **horsepower** have the strongest positive correlation with vehicle price
- **Random Forest** outperformed Linear Regression due to its ability to capture non-linear relationships
- The Logistic Regression classifier successfully distinguished price categories using vehicle characteristics alone

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

1. Clone this repository
2. Place `auto.csv` in the path referenced in the notebook (or update the path in the Load Dataset cell)
3. Open and run the notebook:

```bash
jupyter notebook Predicting_Car_Price_Using_Data_Analytics_and_Machine_Learning.ipynb
```

> **Tip:** Update the dataset path in the `pd.read_csv(...)` cell to match your local environment.

---

## 📈 Results Summary

| Model | Split | R² Score |
|---|---|---|
| Linear Regression | 80/20 | 0.7478805269097566 |
| Random Forest | 80/20 | 0.9283961178869289 (best) |
| Linear Regression | 70/30 | 0.7170204766253411 |
| Random Forest | 70/30 | 0.8800013063077806 |

> Exact metric values are printed when the notebook is run.

---

## 🎓 Learning Outcomes

This project demonstrates:

- **LO1:** Application of linear algebra, statistics, and algorithmic thinking to data-driven problem solving
- **LO2:** Statistical analysis and interpretation to support business decision-making
- **LO3:** Integration of foundational data analytics concepts to address a real-world business challenge

---

## 📄 License

This project is for educational purposes.
