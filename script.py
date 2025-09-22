
"""
MLOps Pipeline for Customer Churn Prediction with CML

This script implements a complete machine learning pipeline for predicting customer churn,
including data preprocessing, model training with different imbalance handling techniques,
and performance evaluation with confusion matrix visualization.

Author: MLOps Team
Date: September 2025
"""

# Standard library imports
import os
from typing import Dict, Any, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Imbalanced learning
from imblearn.over_sampling import SMOTE

# Scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# Configuration constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_STRATEGY = 0.7
N_ESTIMATORS = 100
IQR_FACTOR = 1.5
FIGURE_SIZE = (8, 6)
COMBINED_FIGURE_SIZE = (15, 5)
DPI = 300


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select specific columns from a pandas DataFrame.
    
    This transformer is useful in scikit-learn pipelines when working with
    pandas DataFrames and you need to select specific columns for processing.
    
    Parameters
    ----------
    attribute_names : list
        List of column names to select from the DataFrame
    """
    
    def __init__(self, attribute_names: list) -> None:
        self.attribute_names = attribute_names
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'DataFrameSelector':
        """
        Fit method - no actual fitting required for column selection.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame
        y : array-like, optional
            Target values (ignored)
            
        Returns
        -------
        self : DataFrameSelector
            Returns self for method chaining
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform method to select specified columns.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame
            
        Returns
        -------
        np.ndarray
            Selected columns as numpy array
        """
        return X[self.attribute_names].values


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling outliers using different methods.
    
    Currently supports IQR (Interquartile Range) method for outlier detection
    and capping at the calculated bounds.
    
    Parameters
    ----------
    method : str, default='iqr'
        Method to use for outlier detection ('iqr' is currently supported)
    factor : float, default=1.5
        Factor to multiply IQR for determining outlier bounds
    """
    
    def __init__(self, method: str = "iqr", factor: float = 1.5) -> None:
        self.method = method
        self.factor = factor
        self.lower_bound_: Optional[np.ndarray] = None
        self.upper_bound_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OutlierHandler':
        """
        Fit the outlier handler by calculating bounds.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y : array-like, optional
            Target values (ignored)
            
        Returns
        -------
        self : OutlierHandler
            Returns self for method chaining
        """
        if self.method == "iqr":
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            self.lower_bound_ = Q1 - self.factor * IQR
            self.upper_bound_ = Q3 + self.factor * IQR
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data by capping outliers at the calculated bounds.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
            
        Returns
        -------
        np.ndarray
            Transformed features with outliers capped
        """
        if self.lower_bound_ is None or self.upper_bound_ is None:
            raise ValueError("OutlierHandler must be fitted before transforming")
            
        X_transformed = np.where(X < self.lower_bound_, self.lower_bound_, X)
        X_transformed = np.where(X_transformed > self.upper_bound_, self.upper_bound_, X_transformed)
        return X_transformed


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare the dataset for training.
    
    Returns
    -------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    """
    # Load dataset
    train_path = os.path.join(os.getcwd(), "dataset.csv")
    df = pd.read_csv(train_path)
    
    # Remove unnecessary columns
    df.drop(columns=["CustomerId", "RowNumber", "Surname"], inplace=True)
    
    # Separate features and target
    X = df.drop(columns=['Exited'], axis=1)
    y = df['Exited']
    
    return X, y


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create a comprehensive preprocessing pipeline.
    
    Returns
    -------
    ColumnTransformer
        Complete preprocessing pipeline for all feature types
    """
    # Define column groups
    numerical_cols = ['Age', 'CreditScore', 'Tenure', 'Balance', 'EstimatedSalary'] 
    categorical_cols = ['Geography', 'Gender']  # Assuming these are the categorical columns
    binary_cols = ['NumOfProducts', 'HasCrCard', 'IsActiveMember']
    
    # Numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_handler', OutlierHandler(method='iqr', factor=IQR_FACTOR)),
        ('scaler', RobustScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Binary/ready-to-use pipeline
    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])
    
    # Combine all pipelines
    preprocessing_pipeline = ColumnTransformer(transformers=[
        ('numerical', numerical_pipeline, numerical_cols),
        ('categorical', categorical_pipeline, categorical_cols),
        ('binary', binary_pipeline, binary_cols)
    ])
    
    return preprocessing_pipeline


# Load and prepare data
X, y = load_and_prepare_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Create and apply preprocessing pipeline
preprocessing_pipeline = create_preprocessing_pipeline()
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate balanced class weights for imbalanced dataset.
    
    Parameters
    ----------
    y : np.ndarray
        Target variable
        
    Returns
    -------
    Dict[int, float]
        Dictionary mapping class labels to their weights
    """
    vals_count = np.bincount(y) / len(y)
    
    # Create balanced class weights (inverse of frequency)
    class_weights = {}
    for i in range(len(vals_count)):
        class_weights[i] = 1.0 / vals_count[i] if vals_count[i] > 0 else 1.0
    
    # Normalize weights
    total_weight = sum(class_weights.values())
    for i in range(len(vals_count)):
        class_weights[i] = class_weights[i] / total_weight
    
    return class_weights


def apply_smote_sampling(X: np.ndarray, y: np.ndarray, strategy: float = SMOTE_STRATEGY) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to the training data.
    
    Parameters
    ----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels
    strategy : float, default=0.7
        Sampling strategy for SMOTE
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Resampled features and labels
    """
    smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


# Calculate class weights for imbalanced data handling
class_weights = calculate_class_weights(y_train)

# Apply SMOTE for oversampling
X_train_smote, y_train_smote = apply_smote_sampling(X_train_processed, y_train)



def train_and_evaluate_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = '', 
    class_weight: Optional[Dict[int, float]] = None
) -> str:
    """
    Train a Random Forest model and evaluate its performance.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    experiment_name : str, default=''
        Name of the experiment for logging and plotting
    class_weight : Dict[int, float], optional
        Class weights for handling imbalanced data
        
    Returns
    -------
    str
        The name of the classifier used
    """
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        random_state=RANDOM_STATE, 
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)
    
    # Create and save confusion matrix plot
    _create_confusion_matrix_plot(y_test, y_pred_test, experiment_name)
    
    # Log metrics to file
    _log_metrics_to_file(model.__class__.__name__, experiment_name, f1_train, f1_test)
    
    return model.__class__.__name__


def _create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """
    Create and save a confusion matrix plot.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    title : str
        Title for the plot
    """
    plt.figure(figsize=FIGURE_SIZE)
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap with improved styling
    sns.heatmap(
        cm, 
        annot=True, 
        cbar=False, 
        fmt='d',  # Use integer format for counts
        cmap='Blues',
        square=True,
        linewidths=0.5
    )
    
    plt.title(f'{title}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=['Not Churned', 'Churned'])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=['Not Churned', 'Churned'])
    
    # Save plot with error handling
    try:
        plt.savefig(f'{title}.png', bbox_inches='tight', dpi=DPI)
        plt.close()
    except Exception as e:
        print(f"Error saving plot {title}: {e}")
        plt.close()


def _log_metrics_to_file(model_name: str, experiment_name: str, f1_train: float, f1_test: float) -> None:
    """
    Log model metrics to a text file.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    experiment_name : str
        Name of the experiment
    f1_train : float
        F1 score on training data
    f1_test : float
        F1 score on test data
    """
    try:
        with open('metrics.txt', 'a', encoding='utf-8') as f:
            f.write(f'{model_name} - {experiment_name}\n')
            f.write(f"F1-score Training: {f1_train * 100:.2f}%\n")
            f.write(f"F1-score Validation: {f1_test * 100:.2f}%\n")
            f.write('-' * 50 + '\n')
    except Exception as e:
        print(f"Error writing metrics to file: {e}")


def create_combined_confusion_matrix(model_name: str) -> None:
    """
    Create a combined visualization of all confusion matrices.
    
    Parameters
    ----------
    model_name : str
        Name of the model for the plot title
    """
    confusion_matrix_files = [
        './Without Imbalance Handle.png', 
        './With Imbalance Handle (class_weight).png', 
        './With Imbalance Handle (SMOTE).png'
    ]
    
    # Check if all files exist
    existing_files = [f for f in confusion_matrix_files if os.path.exists(f)]
    
    if not existing_files:
        print("No confusion matrix files found to combine.")
        return
    
    # Create combined plot
    plt.figure(figsize=COMBINED_FIGURE_SIZE)
    
    for i, file_path in enumerate(existing_files, 1):
        try:
            img = Image.open(file_path)
            plt.subplot(1, len(existing_files), i)
            plt.imshow(img)
            plt.axis('off')
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
    
    # Add title and save
    plt.suptitle(f'{model_name} - Comparison of Imbalance Handling Methods', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    try:
        plt.savefig('conf_matrix.png', bbox_inches='tight', dpi=DPI)
        plt.close()
        
        # Clean up individual files
        for file_path in existing_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove file {file_path}: {e}")
                
    except Exception as e:
        print(f"Error saving combined confusion matrix: {e}")
        plt.close()


def main() -> None:
    """
    Main execution function for the MLOps pipeline.
    """
    print("Starting MLOps Pipeline for Customer Churn Prediction...")
    
    # Clear previous metrics file
    if os.path.exists('metrics.txt'):
        os.remove('metrics.txt')
    
    # Train models with different imbalance handling techniques
    print("\n1. Training model without imbalance handling...")
    model_name = train_and_evaluate_model(
        X_train_processed, y_train, X_test_processed, y_test,
        experiment_name='Without Imbalance Handle',
        class_weight=None
    )
    
    print("2. Training model with class weight balancing...")
    train_and_evaluate_model(
        X_train_processed, y_train, X_test_processed, y_test,
        experiment_name='With Imbalance Handle (class_weight)',
        class_weight=class_weights
    )
    
    print("3. Training model with SMOTE oversampling...")
    train_and_evaluate_model(
        X_train_smote, y_train_smote, X_test_processed, y_test,
        experiment_name='With Imbalance Handle (SMOTE)',
        class_weight=None
    )
    
    # Create combined visualization
    print("\n4. Creating combined confusion matrix visualization...")
    create_combined_confusion_matrix(model_name)
    
    print("\nPipeline completed successfully!")
    print("Check 'metrics.txt' for detailed results and 'conf_matrix.png' for visualizations.")


if __name__ == "__main__":
    main()