## `Churn Detection with use of CML & DVC Tools `
    * Using different approaches for solving imbalancing dataset.
    * Using different Algorithms also.
-------------------
### Note
> `cml-churn.yaml` file is attached to this directory. You can put it in `.github/workflows/cml-churn.yaml` as usual.
------------------------

### About

**👨‍💻 Developer:** Ali Ashraf  
**🎯 Role:** Data Scientist / ML Engineer  
**🔬 Specialization:** MLOps, Machine Learning, Data Science  
**📍 Location:** Egypt  

#### 🚀 **Project Overview**
This project demonstrates a complete MLOps pipeline for customer churn prediction, showcasing industry best practices in machine learning operations. The implementation focuses on handling imbalanced datasets through multiple strategies and comparing their effectiveness.

#### 🛠️ **Technologies & Tools Used**
- **Programming:** Python 3.x
- **ML Libraries:** Scikit-learn, Imbalanced-learn, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **MLOps Tools:** CML (Continuous Machine Learning), DVC (Data Version Control)
- **Model:** Random Forest Classifier
- **Preprocessing:** Custom transformers, Pipeline architecture

#### 📊 **Key Features Implemented**
- ✅ Comprehensive data preprocessing pipeline
- ✅ Custom transformer classes for reusable components  
- ✅ Multiple imbalance handling strategies (Class weights, SMOTE)
- ✅ Automated model evaluation and comparison
- ✅ Visual confusion matrix generation
- ✅ Detailed performance metrics logging
- ✅ Clean, modular, and maintainable code structure

#### 🎓 **Learning Source**
This project was developed as part of learning from **Mohamed Agoor's MLOps Course** on YouTube. The course provided excellent guidance on practical MLOps implementation and best practices.

📺 **Course Link:** [Mohamed Agoor YouTube Channel](https://www.youtube.com/@MohamedAgoor)

#### 📈 **Project Impact**
- Demonstrates real-world MLOps pipeline implementation
- Shows effective handling of imbalanced datasets
- Provides reusable components for future ML projects
- Follows industry standards for code organization and documentation

#### 🤝 **Connect & Collaborate**
I'm passionate about Machine Learning, Data Science, and MLOps. Feel free to reach out for discussions, collaborations, or knowledge sharing!

**📧 Email:** [Your Email]  
**💼 LinkedIn:** [Your LinkedIn Profile]  
**🐱 GitHub:** [Your GitHub Profile]  

---
*"Building intelligent systems through data-driven insights and robust ML operations"*

------------------------

### Acknowledgment
This project was developed as part of learning from **Mohamed Agoor's Course** on YouTube. Special thanks for the comprehensive MLOps tutorials and practical guidance.

📺 [Mohamed Agoor YouTube Channel](https://www.youtube.com/@MohammedAgoor)
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


