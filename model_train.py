# Employee Attrition Prediction - Complete ML Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EmployeeAttritionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.model_metrics = {}
        
    def load_and_explore_data(self, data_path=None, df=None):
        """Load and explore the dataset"""
        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.read_csv(data_path)
        
        print("=== DATASET OVERVIEW ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn names: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        return self.df
    
    def visualize_data_exploration(self):
        """Create comprehensive visualizations for data exploration"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Employee Attrition - Data Exploration', fontsize=16, fontweight='bold')
        
        # 1. Distribution of employees who left
        left_counts = self.df['left'].value_counts()
        axes[0,0].pie(left_counts.values, labels=['Stayed', 'Left'], autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Employee Attrition Distribution')
        
        # 2. Satisfaction level vs Working hours (for employees who left)
        left_employees = self.df[self.df['left'] == 1]
        axes[0,1].scatter(left_employees['satisfaction_level'], left_employees['average_montly_hours'], alpha=0.6)
        axes[0,1].set_xlabel('Satisfaction Level')
        axes[0,1].set_ylabel('Average Monthly Hours')
        axes[0,1].set_title('Satisfaction vs Hours (Employees who Left)')
        
        # 3. Department-wise attrition
        dept_attrition = self.df.groupby('Department')['left'].agg(['count', 'sum']).reset_index()
        dept_attrition['attrition_rate'] = (dept_attrition['sum'] / dept_attrition['count']) * 100
        axes[0,2].bar(dept_attrition['Department'], dept_attrition['attrition_rate'])
        axes[0,2].set_xlabel('Department')
        axes[0,2].set_ylabel('Attrition Rate (%)')
        axes[0,2].set_title('Department-wise Attrition Rate')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Salary vs Attrition
        salary_attrition = self.df.groupby('salary')['left'].agg(['count', 'sum']).reset_index()
        salary_attrition['attrition_rate'] = (salary_attrition['sum'] / salary_attrition['count']) * 100
        axes[1,0].bar(salary_attrition['salary'], salary_attrition['attrition_rate'])
        axes[1,0].set_xlabel('Salary Level')
        axes[1,0].set_ylabel('Attrition Rate (%)')
        axes[1,0].set_title('Salary vs Attrition Rate')
        
        # 5. Promotion vs Attrition
        promo_attrition = self.df.groupby('promotion_last_5years')['left'].agg(['count', 'sum']).reset_index()
        promo_attrition['attrition_rate'] = (promo_attrition['sum'] / promo_attrition['count']) * 100
        axes[1,1].bar(['No Promotion', 'Promoted'], promo_attrition['attrition_rate'])
        axes[1,1].set_xlabel('Promotion in Last 5 Years')
        axes[1,1].set_ylabel('Attrition Rate (%)')
        axes[1,1].set_title('Promotion vs Attrition Rate')
        
        # 6. Satisfaction level distribution
        axes[1,2].hist([self.df[self.df['left']==0]['satisfaction_level'], 
                       self.df[self.df['left']==1]['satisfaction_level']], 
                      bins=20, alpha=0.7, label=['Stayed', 'Left'])
        axes[1,2].set_xlabel('Satisfaction Level')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Satisfaction Level Distribution')
        axes[1,2].legend()
        
        # 7. Time spent in company
        axes[2,0].hist([self.df[self.df['left']==0]['time_spend_company'], 
                       self.df[self.df['left']==1]['time_spend_company']], 
                      bins=10, alpha=0.7, label=['Stayed', 'Left'])
        axes[2,0].set_xlabel('Years in Company')
        axes[2,0].set_ylabel('Frequency')
        axes[2,0].set_title('Years in Company Distribution')
        axes[2,0].legend()
        
        # 8. Number of projects
        axes[2,1].hist([self.df[self.df['left']==0]['number_project'], 
                       self.df[self.df['left']==1]['number_project']], 
                      bins=8, alpha=0.7, label=['Stayed', 'Left'])
        axes[2,1].set_xlabel('Number of Projects')
        axes[2,1].set_ylabel('Frequency')
        axes[2,1].set_title('Number of Projects Distribution')
        axes[2,1].legend()
        
        # 9. Last evaluation score
        axes[2,2].hist([self.df[self.df['left']==0]['last_evaluation'], 
                       self.df[self.df['left']==1]['last_evaluation']], 
                      bins=20, alpha=0.7, label=['Stayed', 'Left'])
        axes[2,2].set_xlabel('Last Evaluation Score')
        axes[2,2].set_ylabel('Frequency')
        axes[2,2].set_title('Last Evaluation Distribution')
        axes[2,2].legend()
        
        plt.tight_layout()
        plt.show()
        
    def generate_insights(self):
        """Generate detailed insights from the data"""
        print("\n=== KEY INSIGHTS FROM DATA EXPLORATION ===")
        
        # Overall attrition rate
        attrition_rate = (self.df['left'].sum() / len(self.df)) * 100
        print(f"1. Overall Attrition Rate: {attrition_rate:.1f}%")
        
        # Satisfaction level insights
        left_satisfaction = self.df[self.df['left']==1]['satisfaction_level'].mean()
        stayed_satisfaction = self.df[self.df['left']==0]['satisfaction_level'].mean()
        print(f"2. Average Satisfaction Level:")
        print(f"   - Employees who left: {left_satisfaction:.2f}")
        print(f"   - Employees who stayed: {stayed_satisfaction:.2f}")
        
        # Working hours insights
        left_hours = self.df[self.df['left']==1]['average_montly_hours'].mean()
        stayed_hours = self.df[self.df['left']==0]['average_montly_hours'].mean()
        print(f"3. Average Monthly Hours:")
        print(f"   - Employees who left: {left_hours:.1f} hours")
        print(f"   - Employees who stayed: {stayed_hours:.1f} hours")
        
        # Department insights
        dept_attrition = self.df.groupby('Department')['left'].agg(['count', 'sum']).reset_index()
        dept_attrition['attrition_rate'] = (dept_attrition['sum'] / dept_attrition['count']) * 100
        highest_attrition_dept = dept_attrition.loc[dept_attrition['attrition_rate'].idxmax()]
        print(f"4. Highest Attrition Department: {highest_attrition_dept['Department']} ({highest_attrition_dept['attrition_rate']:.1f}%)")
        
        # Salary insights
        salary_attrition = self.df.groupby('salary')['left'].agg(['count', 'sum']).reset_index()
        salary_attrition['attrition_rate'] = (salary_attrition['sum'] / salary_attrition['count']) * 100
        print(f"5. Attrition by Salary Level:")
        for _, row in salary_attrition.iterrows():
            print(f"   - {row['salary'].title()}: {row['attrition_rate']:.1f}%")
        
        # Promotion insights
        promo_attrition = self.df.groupby('promotion_last_5years')['left'].agg(['count', 'sum']).reset_index()
        promo_attrition['attrition_rate'] = (promo_attrition['sum'] / promo_attrition['count']) * 100
        print(f"6. Promotion Impact:")
        print(f"   - No promotion: {promo_attrition[promo_attrition['promotion_last_5years']==0]['attrition_rate'].iloc[0]:.1f}% attrition")
        print(f"   - Promoted: {promo_attrition[promo_attrition['promotion_last_5years']==1]['attrition_rate'].iloc[0]:.1f}% attrition")
        
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        # Create a copy for preprocessing
        self.processed_df = self.df.copy()
        
        # Handle categorical variables
        categorical_columns = ['Department', 'salary']
        
        for col in categorical_columns:
            le = LabelEncoder()
            self.processed_df[col] = le.fit_transform(self.processed_df[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        self.X = self.processed_df.drop(['left'], axis=1)
        self.y = self.processed_df['left']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Data preprocessing completed!")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
    def train_models(self):
        """Train multiple models and select the best one"""
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        best_score = 0
        best_model_name = ""
        
        print("\n=== MODEL TRAINING AND EVALUATION ===")
        
        for name, model in models.items():
            if name == 'Logistic Regression':
                # Use scaled data for logistic regression
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                # Use original data for tree-based models
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            if name == 'Logistic Regression':
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
            
            print(f"\n{name}:")
            print(f"  AUC Score: {auc_score:.4f}")
            print(f"  CV AUC Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store metrics
            self.model_metrics[name] = {
                'auc_score': auc_score,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Select best model
            if auc_score > best_score:
                best_score = auc_score
                best_model_name = name
                self.model = model
                self.best_predictions = y_pred
                self.best_predictions_proba = y_pred_proba
        
        print(f"\n*** Best Model: {best_model_name} with AUC Score: {best_score:.4f} ***")
        
        # Fine-tune the best model if it's Random Forest
        if best_model_name == 'Random Forest':
            self.fine_tune_random_forest()
        
    def fine_tune_random_forest(self):
        """Fine-tune Random Forest hyperparameters"""
        print("\nFine-tuning Random Forest...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Update the model with best parameters
        self.model = grid_search.best_estimator_
        self.best_predictions = self.model.predict(self.X_test)
        self.best_predictions_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
    def evaluate_model(self):
        """Evaluate the final model"""
        print("\n=== FINAL MODEL EVALUATION ===")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.best_predictions))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.best_predictions)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # AUC Score
        final_auc = roc_auc_score(self.y_test, self.best_predictions_proba)
        print(f"\nFinal AUC Score: {final_auc:.4f}")
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
        
    def visualize_results(self):
        """Create visualizations for model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.best_predictions_proba)
        auc_score = roc_auc_score(self.y_test, self.best_predictions_proba)
        axes[0,0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0,0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[0,0].set_xlabel('False Positive Rate')
        axes[0,0].set_ylabel('True Positive Rate')
        axes[0,0].set_title('ROC Curve')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 2. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        axes[0,1].set_title('Confusion Matrix')
        
        # 3. Feature Importance (if available)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1,0].barh(range(len(top_features)), top_features['importance'])
            axes[1,0].set_yticks(range(len(top_features)))
            axes[1,0].set_yticklabels(top_features['feature'])
            axes[1,0].set_xlabel('Importance')
            axes[1,0].set_title('Top 10 Feature Importances')
            axes[1,0].invert_yaxis()
        
        # 4. Model Comparison
        model_names = list(self.model_metrics.keys())
        auc_scores = [self.model_metrics[name]['auc_score'] for name in model_names]
        axes[1,1].bar(model_names, auc_scores)
        axes[1,1].set_xlabel('Models')
        axes[1,1].set_ylabel('AUC Score')
        axes[1,1].set_title('Model Comparison')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def generate_model_insights(self):
        """Generate insights from the trained model"""
        print("\n=== MODEL INSIGHTS AND RECOMMENDATIONS ===")
        
        if self.feature_importance is not None:
            print("1. TOP FACTORS CONTRIBUTING TO EMPLOYEE ATTRITION:")
            for i, row in self.feature_importance.head(5).iterrows():
                feature_name = row['feature']
                importance = row['importance']
                
                # Provide interpretable names
                interpretable_names = {
                    'satisfaction_level': 'Employee Satisfaction Level',
                    'time_spend_company': 'Years Spent in Company',
                    'average_montly_hours': 'Average Monthly Working Hours',
                    'last_evaluation': 'Last Performance Evaluation Score',
                    'number_project': 'Number of Projects Assigned',
                    'Work_accident': 'Workplace Accident History',
                    'promotion_last_5years': 'Promotion in Last 5 Years',
                    'Department': 'Department',
                    'salary': 'Salary Level'
                }
                
                readable_name = interpretable_names.get(feature_name, feature_name)
                print(f"   - {readable_name}: {importance:.3f}")
        
        print("\n2. KEY RECOMMENDATIONS:")
        
        # Satisfaction level insights
        left_satisfaction = self.df[self.df['left']==1]['satisfaction_level'].mean()
        if left_satisfaction < 0.5:
            print("   - Focus on improving employee satisfaction through better work-life balance")
            print("   - Implement regular satisfaction surveys and feedback mechanisms")
        
        # Working hours insights
        left_hours = self.df[self.df['left']==1]['average_montly_hours'].mean()
        stayed_hours = self.df[self.df['left']==0]['average_montly_hours'].mean()
        if left_hours > stayed_hours:
            print("   - Monitor and control excessive working hours")
            print("   - Consider workload redistribution for overworked employees")
        
        # Promotion insights
        promo_attrition = self.df.groupby('promotion_last_5years')['left'].mean()
        if promo_attrition[0] > promo_attrition[1]:
            print("   - Develop clear career advancement paths")
            print("   - Increase promotion opportunities for deserving employees")
        
        # Salary insights
        salary_attrition = self.df.groupby('salary')['left'].mean()
        if 'low' in salary_attrition.index and salary_attrition['low'] > 0.3:
            print("   - Review and adjust compensation packages, especially for lower salary tiers")
            print("   - Implement performance-based salary increments")
        
        print("\n3. RISK PREDICTION:")
        print("   - Use this model to identify employees at high risk of leaving")
        print("   - Implement proactive retention strategies for high-risk employees")
        print("   - Schedule regular check-ins with employees scoring >0.7 on the attrition probability")
        
    def save_model(self, model_path='employee_attrition_model.joblib', scaler_path='scaler.joblib'):
        """Save the trained model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, 'label_encoders.joblib')
        
        print(f"\n=== MODEL SAVED ===")
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Label encoders saved to: label_encoders.joblib")
        
        # Save model summary
        model_summary = {
            'model_type': type(self.model).__name__,
            'features': list(self.X.columns),
            'auc_score': roc_auc_score(self.y_test, self.best_predictions_proba),
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None
        }
        joblib.dump(model_summary, 'model_summary.joblib')
        print(f"Model summary saved to: model_summary.joblib")
        
    def predict_attrition(self, employee_data):
        """Predict attrition for new employee data"""
        if isinstance(employee_data, dict):
            employee_df = pd.DataFrame([employee_data])
        else:
            employee_df = employee_data.copy()
        
        # Apply same preprocessing
        for col, encoder in self.label_encoders.items():
            if col in employee_df.columns:
                employee_df[col] = encoder.transform(employee_df[col])
        
        # Scale if using logistic regression, otherwise use original
        if isinstance(self.model, LogisticRegression):
            employee_scaled = self.scaler.transform(employee_df)
            prediction = self.model.predict(employee_scaled)
            probability = self.model.predict_proba(employee_scaled)[:, 1]
        else:
            prediction = self.model.predict(employee_df)
            probability = self.model.predict_proba(employee_df)[:, 1]
        
        return prediction, probability

# Example usage and demonstration
def main():
    # Create sample data (replace with your actual data loading)
    np.random.seed(42)
    
    # Since you provided a sample of your data, I'll create a synthetic dataset similar to yours
    n_samples = 15000
    
    data = {
        'satisfaction_level': np.random.beta(2, 2, n_samples),
        'last_evaluation': np.random.beta(2, 2, n_samples),
        'number_project': np.random.choice([2, 3, 4, 5, 6, 7], n_samples, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]),
        'average_montly_hours': np.random.normal(200, 50, n_samples).astype(int),
        'time_spend_company': np.random.choice(range(2, 11), n_samples),
        'Work_accident': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'promotion_last_5years': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'Department': np.random.choice(['sales', 'technical', 'support', 'IT', 'marketing', 'product_mng', 'RandD', 'accounting', 'hr', 'management'], n_samples),
        'salary': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.5, 0.4, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on realistic patterns
    left_prob = (
        0.5 * (1 - df['satisfaction_level']) +  # Lower satisfaction = higher attrition
        0.2 * (df['average_montly_hours'] > 250).astype(int) +  # Overwork
        0.2 * (df['time_spend_company'] > 6).astype(int) +  # Long tenure
        0.1 * (df['salary'] == 'low').astype(int) -  # Low salary
        0.3 * df['promotion_last_5years']  # Recent promotion reduces attrition
    )
    
    df['left'] = np.random.binomial(1, np.clip(left_prob, 0, 1), n_samples)
    
    # Initialize the predictor
    predictor = EmployeeAttritionPredictor()
    
    # Load and explore data
    predictor.load_and_explore_data(r"C:\\Users\\ABHIRAM YADAV M\\Projects\\Attrition_analysis\\HR_comma_sep.csv")
    
    # Generate insights
    predictor.generate_insights()
    
    # Visualize data exploration
    predictor.visualize_data_exploration()
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Train models
    predictor.train_models()
    
    # Evaluate the best model
    predictor.evaluate_model()
    
    # Visualize results
    predictor.visualize_results()
    
    # Generate model insights
    predictor.generate_model_insights()
    
    # Save the model
    predictor.save_model()
    
    # Example prediction for a new employee
    new_employee = {
        'satisfaction_level': 0.3,
        'last_evaluation': 0.8,
        'number_project': 6,
        'average_montly_hours': 280,
        'time_spend_company': 4,
        'Work_accident': 0,
        'promotion_last_5years': 0,
        'Department': 'sales',
        'salary': 'low'
    }
    
    prediction, probability = predictor.predict_attrition(new_employee)
    print(f"\n=== EXAMPLE PREDICTION ===")
    print(f"Employee will leave: {'Yes' if prediction[0] == 1 else 'No'}")
    print(f"Attrition probability: {probability[0]:.3f}")
    
    return predictor

if __name__ == "__main__":
    # Run the complete pipeline
    predictor = main()