## `Churn Detection with use of CML & DVC Tools `
    * Using different approaches for solving imbalancing dataset.
    * Using different Algorithms also.
-------------------
### Note
> `cml-churn.yaml` file is attached to this directory. You can put it in `.github/workflows/cml-churn.yaml` as usual.
------------------------

### Acknowledgment
This project was developed as part of learning from **Mohamed Agoor's Course** on YouTube. Special thanks for the comprehensive MLOps tutorials and practical guidance.

📺 [Mohamed Agoor YouTube Channel](https://www.youtube.com/@MohammedAgoor)
------------------------

### About
**Developer:** Ali Ashraf 
**Role:** Data Scientist / ML Engineer  
**Focus:** MLOps, Machine Learning, Data Science  

This project demonstrates practical implementation of MLOps concepts including:
- Data preprocessing and feature engineering
- Handling imbalanced datasets
- Model training and evaluation
- CI/CD with CML (Continuous Machine Learning)
- Version control with DVC (Data Version Control)

Feel free to connect and discuss ML/MLOps topics!
------------------------

### What We Built in the Script

Our `script.py` implements a comprehensive Machine Learning pipeline for customer churn prediction with the following key components:

#### 🔧 **Custom Transformers**
- **DataFrameSelector**: Custom transformer for selecting specific columns in sklearn pipelines
- **OutlierHandler**: IQR-based outlier detection and clipping transformer

#### 📊 **Data Preprocessing Pipeline**
- **Numerical Features**: Median imputation → Outlier handling → Robust scaling
- **Categorical Features**: Mode imputation → One-hot encoding
- **Ready Features**: Simple imputation for binary/numeric ready-to-use features

#### ⚖️ **Class Imbalance Handling**
- **Method 1**: No imbalance handling (baseline)
- **Method 2**: Class weights (inverse frequency weighting)
- **Method 3**: SMOTE (Synthetic Minority Oversampling Technique)

#### 🤖 **Model Training & Evaluation**
- **Algorithm**: Logistic Regression with different imbalance strategies
- **Metrics**: F1-score for training and validation sets
- **Visualization**: Confusion matrices for each approach
- **Output**: Combined confusion matrix plot and detailed metrics file

#### 📈 **Results & Comparison**
- Automated comparison of three different approaches to handle class imbalance
- Visual confusion matrices saved as PNG files
- Comprehensive metrics logging to `metrics.txt`
- Clean, organized output with automatic file cleanup

The script demonstrates best practices in MLOps including modular code design, proper preprocessing pipelines, and comprehensive model evaluation.


