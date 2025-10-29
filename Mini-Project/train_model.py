#!/usr/bin/env python3
"""
Loan Prediction Analysis - Model Training Script
Comprehensive data-driven framework for loan approval prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class LoanPredictionModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_and_explore_data(self, filepath='train.csv'):
        """Load and perform initial data exploration"""
        print("Loading and exploring data...")
        self.df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nDataset info:")
        print(self.df.info())
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        print(f"\nTarget variable distribution:")
        print(self.df['Loan_Status'].value_counts())
        
        return self.df
    
    def preprocess_data(self):
        """Comprehensive data preprocessing and feature engineering"""
        print("\nPreprocessing data...")
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Handle missing values
        # Categorical variables - fill with mode
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        # Numerical variables - fill with median
        numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for col in numerical_cols:
            if col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Feature Engineering
        # Total Income
        df_processed['TotalIncome'] = df_processed['ApplicantIncome'] + df_processed['CoapplicantIncome']
        
        # Income to Loan Amount Ratio
        df_processed['Income_LoanAmount_Ratio'] = df_processed['TotalIncome'] / (df_processed['LoanAmount'] + 1)
        
        # Loan Amount per Term
        df_processed['LoanAmount_per_Term'] = df_processed['LoanAmount'] / (df_processed['Loan_Amount_Term'] + 1)
        
        # Log transformations for skewed features
        df_processed['Log_ApplicantIncome'] = np.log1p(df_processed['ApplicantIncome'])
        df_processed['Log_TotalIncome'] = np.log1p(df_processed['TotalIncome'])
        df_processed['Log_LoanAmount'] = np.log1p(df_processed['LoanAmount'])
        
        # Encode categorical variables
        categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        
        for feature in categorical_features:
            if feature in df_processed.columns:
                le = LabelEncoder()
                df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
                self.label_encoders[feature] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        df_processed['Loan_Status'] = le_target.fit_transform(df_processed['Loan_Status'])
        self.label_encoders['Loan_Status'] = le_target
        
        # Select features for modeling
        feature_columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area', 'TotalIncome', 'Income_LoanAmount_Ratio',
            'LoanAmount_per_Term', 'Log_ApplicantIncome', 'Log_TotalIncome', 'Log_LoanAmount'
        ]
        
        self.X = df_processed[feature_columns]
        self.y = df_processed['Loan_Status']
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=feature_columns)
        
        print(f"Processed dataset shape: {self.X.shape}")
        print("Data preprocessing completed!")
        
        return self.X, self.y
    
    def train_models(self):
        """Train multiple models and compare performance"""
        print("\nTraining models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            model_scores[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            if auc_score:
                print(f"AUC Score: {auc_score:.4f}")
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['test_accuracy'])
        self.best_model = model_scores[best_model_name]['model']
        self.models = model_scores
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Test Accuracy: {model_scores[best_model_name]['test_accuracy']:.4f}")
        
        # Feature importance for Random Forest
        if 'Random Forest' in model_scores:
            rf_model = model_scores['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = feature_importance
            print(f"\nTop 10 Important Features:")
            print(feature_importance.head(10))
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        return model_scores
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations for analysis"""
        print("\nGenerating visualizations...")
        
        # Create static directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target Distribution
        plt.subplot(3, 4, 1)
        loan_status_counts = self.df['Loan_Status'].value_counts()
        plt.pie(loan_status_counts.values, labels=loan_status_counts.index, autopct='%1.1f%%')
        plt.title('Loan Status Distribution')
        
        # 2. Income Distribution
        plt.subplot(3, 4, 2)
        plt.hist(self.df['ApplicantIncome'], bins=50, alpha=0.7, label='Applicant Income')
        plt.hist(self.df['CoapplicantIncome'], bins=50, alpha=0.7, label='Coapplicant Income')
        plt.xlabel('Income')
        plt.ylabel('Frequency')
        plt.title('Income Distribution')
        plt.legend()
        plt.yscale('log')
        
        # 3. Loan Amount vs Income
        plt.subplot(3, 4, 3)
        approved = self.df[self.df['Loan_Status'] == 'Y']
        rejected = self.df[self.df['Loan_Status'] == 'N']
        plt.scatter(approved['ApplicantIncome'], approved['LoanAmount'], alpha=0.6, label='Approved', s=20)
        plt.scatter(rejected['ApplicantIncome'], rejected['LoanAmount'], alpha=0.6, label='Rejected', s=20)
        plt.xlabel('Applicant Income')
        plt.ylabel('Loan Amount')
        plt.title('Loan Amount vs Income')
        plt.legend()
        
        # 4. Credit History Impact
        plt.subplot(3, 4, 4)
        credit_loan = pd.crosstab(self.df['Credit_History'], self.df['Loan_Status'], normalize='index') * 100
        credit_loan.plot(kind='bar', ax=plt.gca())
        plt.title('Credit History vs Loan Approval')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        
        # 5. Property Area Analysis
        plt.subplot(3, 4, 5)
        property_loan = pd.crosstab(self.df['Property_Area'], self.df['Loan_Status'], normalize='index') * 100
        property_loan.plot(kind='bar', ax=plt.gca())
        plt.title('Property Area vs Loan Approval')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        
        # 6. Education Impact
        plt.subplot(3, 4, 6)
        education_loan = pd.crosstab(self.df['Education'], self.df['Loan_Status'], normalize='index') * 100
        education_loan.plot(kind='bar', ax=plt.gca())
        plt.title('Education vs Loan Approval')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        
        # 7. Model Comparison
        plt.subplot(3, 4, 7)
        if hasattr(self, 'models'):
            model_names = list(self.models.keys())
            accuracies = [self.models[name]['test_accuracy'] for name in model_names]
            plt.bar(model_names, accuracies)
            plt.title('Model Accuracy Comparison')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            for i, v in enumerate(accuracies):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 8. Feature Importance
        plt.subplot(3, 4, 8)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance')
        
        # 9. Confusion Matrix
        plt.subplot(3, 4, 9)
        if hasattr(self, 'models') and self.best_model is not None:
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['test_accuracy'])
            y_pred = self.models[best_model_name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=plt.gca())
            plt.title(f'Confusion Matrix - {best_model_name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
        
        # 10. ROC Curve
        plt.subplot(3, 4, 10)
        if hasattr(self, 'models'):
            for name, model_info in self.models.items():
                if model_info['y_pred_proba'] is not None:
                    fpr, tpr, _ = roc_curve(self.y_test, model_info['y_pred_proba'])
                    auc_score = model_info['auc_score']
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
        
        # 11. Loan Amount Distribution by Status
        plt.subplot(3, 4, 11)
        self.df.boxplot(column='LoanAmount', by='Loan_Status', ax=plt.gca())
        plt.title('Loan Amount Distribution by Status')
        plt.suptitle('')
        
        # 12. Correlation Heatmap
        plt.subplot(3, 4, 12)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, ax=plt.gca())
        plt.title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('static/loan_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print("Visualizations saved to static/loan_analysis_dashboard.png")
    
    def save_model(self, filename='loan_model.joblib'):
        """Save the trained model and preprocessing objects"""
        if self.best_model is not None:
            model_package = {
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': list(self.X.columns),
                'feature_importance': self.feature_importance
            }
            joblib.dump(model_package, filename)
            print(f"Model saved to {filename}")
        else:
            print("No trained model to save!")

def main():
    """Main training pipeline"""
    print("=== Loan Prediction Analysis - Model Training ===")
    
    # Initialize model
    loan_model = LoanPredictionModel()
    
    # Load and explore data
    df = loan_model.load_and_explore_data('train.csv')
    
    # Preprocess data
    X, y = loan_model.preprocess_data()
    
    # Train models
    model_scores = loan_model.train_models()
    
    # Generate visualizations
    loan_model.generate_visualizations()
    
    # Save model
    loan_model.save_model('loan_model.joblib')
    
    print("\n=== Training Complete ===")
    print("Model and visualizations are ready for the web application!")
    print("Run 'python3 backend.py' to start the dashboard!")

if __name__ == "__main__":
    main()