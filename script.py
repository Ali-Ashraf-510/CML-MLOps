"""
Machine Learning Pipeline for Customer Churn Prediction
========================================================
This script implements a complete ML pipeline including data preprocessing,
handling class imbalance, model training, and evaluation using Random Forest.
"""

# =====================================================================================
# IMPORTS AND DEPENDENCIES
# =====================================================================================

## Core libraries for data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
from PIL import Image

## Scikit-learn preprocessing modules
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, FeatureUnion

## Scikit-learn machine learning models
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

## Scikit-learn evaluation metrics
from sklearn.metrics import f1_score, confusion_matrix

# =====================================================================================
# CUSTOM TRANSFORMER CLASSES
# =====================================================================================

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select specific columns from a DataFrame.
    Used in sklearn pipelines to extract specific feature subsets.
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle outliers using IQR method.
    Clips values outside the IQR bounds to reduce impact of extreme values.
    """
    def __init__(self, method="iqr", factor=1.5):
        self.method = method
        self.factor = factor
    
    def fit(self, X, y=None):
        # Calculate IQR bounds for outlier detection
        if self.method == "iqr":
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            self.lower_bound_ = Q1 - self.factor * IQR
            self.upper_bound_ = Q3 + self.factor * IQR
        return self
    
    def transform(self, X):
        # Clip outliers to the calculated bounds
        X = np.where(X < self.lower_bound_, self.lower_bound_, X)
        X = np.where(X > self.upper_bound_, self.upper_bound_, X)
        return X

# =====================================================================================
# DATA LOADING AND INITIAL PREPROCESSING
# =====================================================================================

# Load the dataset
TRAIN_PATH = os.path.join(os.getcwd(), "dataset.csv")
df = pd.read_csv(TRAIN_PATH)

# Remove unnecessary columns that don't contribute to prediction
df.drop(columns=["CustomerId" , "RowNumber" , "Surname"], inplace=True)

# =====================================================================================
# FEATURE ENGINEERING AND DATA PREPARATION
# =====================================================================================

# Separate features and target variable
X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']

# Split data into training and testing sets with stratification to maintain class balance
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

# Define column types for different preprocessing approaches
categ_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = ['Age', 'CreditScore', 'Tenure', 'Balance' , 'EstimatedSalary'] 
ready_cols = ['NumOfProducts' ,'HasCrCard' ,'IsActiveMember']



## Slice the lists
num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']

# =====================================================================================
# PREPROCESSING PIPELINES
# =====================================================================================

# Pipeline for numerical features: imputation, outlier handling, and scaling
num_pipeline = Pipeline(steps = [
    ('imputer' , SimpleImputer(strategy='median')),           # Fill missing values with median
    ('outlier_handler' , OutlierHandler(method='iqr', factor=1.5)),  # Handle outliers
    ('scaler' , RobustScaler())                               # Scale features (robust to outliers)
])

# Pipeline for categorical features: imputation and encoding
categ_pipline = Pipeline(steps=[
    ('imputer' , SimpleImputer(strategy='most_frequent')),    # Fill missing values with mode
    ("ohe" , OneHotEncoder(handle_unknown='ignore' , drop='first'))  # One-hot encode categories
])

# Pipeline for ready-to-use features: only imputation needed
ready_pipeline = Pipeline(steps=[
    ('imputer' , SimpleImputer(strategy='most_frequent')),    # Fill missing values with mode
])

# Combine all preprocessing pipelines
from sklearn.compose import ColumnTransformer
all_pipeline = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', categ_pipline, categ_cols),
    ('ready', ready_pipeline, ready_cols)
])

# Apply preprocessing to training and test data
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)

# =====================================================================================
# CLASS IMBALANCE HANDLING
# =====================================================================================

# Calculate class distribution for balanced weights
vals_count = np.bincount(y_train) / len(y_train)

# Create balanced class weights (inverse of frequency)
dict_wieght = {}
for i in range(2):
    dict_wieght[i] = 1.0 / vals_count[i] if vals_count[i] > 0 else 1.0

# Normalize weights to sum to 1
total_weight = sum(dict_wieght.values())
for i in range(2):
    dict_wieght[i] = dict_wieght[i] / total_weight

# Apply SMOTE for synthetic minority oversampling
over = SMOTE(sampling_strategy=0.7)
X_train_resampled, y_train_resampled = over.fit_resample(X_train_final, y_train)

# =====================================================================================
# MODEL TRAINING AND EVALUATION FUNCTION
# =====================================================================================

def train_model(X_train , y_train , plot_name = '', class_weight = None ):
    """
    Train a Random Forest model and evaluate its performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        plot_name: Name for saving confusion matrix plot
        class_weight: Dictionary of class weights for handling imbalance
    
    Returns:
        bool: True when training and evaluation complete
    """
    global clf_name
    
    # Initialize and train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
    clf.fit(X_train , y_train)
    
    # Make predictions on training and test sets
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test_final)
    
    # Calculate F1 scores for both sets
    f1_score_train = f1_score(y_train , y_pred_train)
    f1_score_test = f1_score(y_test , y_pred_test)
    clf_name = clf.__class__.__name__
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Log performance metrics to file
    with open('metrics.txt', 'a') as f:
        f.write(f'{clf_name} {plot_name}\n')
        f.write(f"F1-score of Training is: {f1_score_train*100:.2f} %\n")
        f.write(f"F1-Score of Validation is: {f1_score_test*100:.2f} %\n")
        f.write('----'*10 + '\n')
    return True

# =====================================================================================
# MODEL TRAINING WITH DIFFERENT IMBALANCE HANDLING STRATEGIES
# =====================================================================================

# Strategy 1: Train model without addressing class imbalance
train_model(X_train_final , y_train , plot_name = 'Without Imbalance Handle' , class_weight = None)

# Strategy 2: Train model using class weights to handle imbalance
train_model(X_train_final , y_train , plot_name = 'With Imbalance Handle (class_weight)' , class_weight = dict_wieght)

# Strategy 3: Train model using SMOTE oversampling to handle imbalance
train_model(X_train_resampled , y_train_resampled , plot_name = 'With Imbalance Handle (SMOTE)' , class_weight = None)

# =====================================================================================
# RESULTS VISUALIZATION AND CLEANUP
# =====================================================================================

# Combine all confusion matrix plots into a single visualization
confusion_matrix_paths = ['./Without Imbalance Handle.png', './With Imbalance Handle (class_weight).png', './With Imbalance Handle (SMOTE).png']

# Load and arrange all confusion matrices in a single plot
plt.figure(figsize=(15, 5))  # Adjust figure size as needed
for i, path in enumerate(confusion_matrix_paths, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(img)
    plt.axis('off')  # Disable axis for cleaner visualization

# Save the combined confusion matrix plot
plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'conf_matrix.png', bbox_inches='tight', dpi=300)

# Clean up individual confusion matrix files
for path in confusion_matrix_paths:
    os.remove(path)
