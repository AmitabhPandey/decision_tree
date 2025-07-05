# 🧾 Fake Banknote Detection using Decision Tree Classifier

This project implements a **Decision Tree Classifier from scratch** in Python to **detect fake vs genuine banknotes** using features extracted from banknote images.

The features in the dataset were obtained by applying **wavelet transform** on scanned images of banknotes, capturing various statistical properties. The goal is to accurately classify whether a banknote is authentic (`1`) or counterfeit (`0`) based on these features.

---

**Problem Statement**: Build a decision tree classifier to distinguish between real and fake banknotes.

**Dataset Source**: UCI Machine Learning Repository - Banknote Authentication Dataset

**Features**:

- `variance`: Wavelet transformed image (continuous)
- `skewness`: of the image
- `kurtosis`: of the image
- `entropy`: of the image

**Label**:

- `0` – Counterfeit
- `1` – Genuine

---

### 🔧 Features Implemented

- Custom Decision Tree Classifier (built from scratch)
- Gini Index as the splitting criterion
- Tree building with configurable max depth and min samples per node
- Tree traversal for predictions
- Performance metrics: Accuracy, Confusion Matrix, Classification Report
- Feature importance analysis
- Pretty-printed tree structure in the console

### 🧪 Methodology

- Load and preprocess the dataset (`data_banknote_authentication.csv`).
- Split into training (70%) and testing (30%).
- Build the tree using Gini Index as the impurity measure.
- Stop tree growth using parameters like max depth and min elements per node.
- Predict on both training and testing datasets.
- Evaluate using accuracy and scikit-learn metrics.
- Print feature importance and tree structure.

### 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/fake-banknote-detection.git
cd fake-banknote-detection
```

2. Run the Jupyter notebook `Banknote_DecisionTree.ipynb` in your preferred environment (Jupyter/Colab).

### 📈 Sample Output

```
Train Accuracy: 98.73%
Test Accuracy: 96.64%
Feature Importance:
  Feature 0: 12 splits
  Feature 2: 7 splits
  Feature 1: 4 splits
  Feature 3: 2 splits
```

### 🌳 Tree Structure Preview

```
[X0 <= 2.3857]
--> True:
    [X1 <= 6.2345]
    --> True:
        Leaf: Predict 0
    --> False:
        Leaf: Predict 1
--> False:
    Leaf: Predict 1
```

---

**Tech Stack**:

- Python 3.8+
- NumPy
- Pandas
- scikit-learn (for evaluation metrics)
- Jupyter Notebook

**References**:

- [UCI ML Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

**Author**: Amitabh Pandey
