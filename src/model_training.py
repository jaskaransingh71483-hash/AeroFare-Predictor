"""
Model Training Module for Flight Fare Prediction
Trains and evaluates multiple regression models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class FlightFarePredictor:
    """
    Flight Fare Prediction Model Trainer
    Supports multiple regression algorithms
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, df, target_column='Price', test_size=0.2):
        """Split data into train and test sets"""
        print("\nPreparing data for modeling...")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Feature columns: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize all models to be trained"""
        print("\nInitializing models...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=15,
                min_samples_split=10,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                random_state=self.random_state
            )
        }
        
        print(f"Models initialized: {list(self.models.keys())}")
        return self.models
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate a single model"""
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Store results
        results = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_test_pred
        }
        
        return results
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.initialize_models()
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            results = self.evaluate_model(
                model, X_train, X_test, y_train, y_test, model_name
            )
            self.results[model_name] = results
            
            print(f"  Train R²: {results['train_r2']:.4f}")
            print(f"  Test R²: {results['test_r2']:.4f}")
            print(f"  Test RMSE: ₹{results['test_rmse']:.2f}")
            print(f"  Test MAE: ₹{results['test_mae']:.2f}")
        
        # Find best model based on test R²
        self.best_model_name = max(
            self.results.items(),
            key=lambda x: x[1]['test_r2']
        )[0]
        self.best_model = self.results[self.best_model_name]['model']
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Test R² Score: {self.results[self.best_model_name]['test_r2']:.4f}")
        print("="*60)
        
        return self.results
    
    def get_comparison_dataframe(self):
        """Get comparison of all models as DataFrame"""
        comparison = []
        
        for model_name, results in self.results.items():
            comparison.append({
                'Model': model_name,
                'Train R²': results['train_r2'],
                'Test R²': results['test_r2'],
                'Train RMSE': results['train_rmse'],
                'Test RMSE': results['test_rmse'],
                'Train MAE': results['train_mae'],
                'Test MAE': results['test_mae']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('Test R²', ascending=False)
        
        return df_comparison
    
    def plot_model_comparison(self, save_path=None):
        """Plot comparison of model performances"""
        df_comparison = self.get_comparison_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # R² Score comparison
        ax1 = axes[0, 0]
        x = np.arange(len(df_comparison))
        width = 0.35
        ax1.bar(x - width/2, df_comparison['Train R²'], width, label='Train R²', alpha=0.8)
        ax1.bar(x + width/2, df_comparison['Test R²'], width, label='Test R²', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # RMSE comparison
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, df_comparison['Train RMSE'], width, label='Train RMSE', alpha=0.8)
        ax2.bar(x + width/2, df_comparison['Test RMSE'], width, label='Test RMSE', alpha=0.8)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('RMSE (₹)')
        ax2.set_title('RMSE Comparison (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # MAE comparison
        ax3 = axes[1, 0]
        ax3.bar(x - width/2, df_comparison['Train MAE'], width, label='Train MAE', alpha=0.8)
        ax3.bar(x + width/2, df_comparison['Test MAE'], width, label='Test MAE', alpha=0.8)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('MAE (₹)')
        ax3.set_title('MAE Comparison (Lower is Better)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Test R² Score ranking
        ax4 = axes[1, 1]
        colors = ['green' if model == self.best_model_name else 'steelblue' 
                  for model in df_comparison['Model']]
        ax4.barh(df_comparison['Model'], df_comparison['Test R²'], color=colors, alpha=0.8)
        ax4.set_xlabel('Test R² Score')
        ax4.set_title('Model Ranking by Test R²')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        return fig
    
    def save_best_model(self, filepath):
        """Save the best model to disk"""
        if self.best_model is None:
            print("No model trained yet!")
            return
        
        joblib.dump(self.best_model, filepath)
        print(f"\nBest model ({self.best_model_name}) saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.best_model
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model trained or loaded!")
        
        return self.best_model.predict(X)


if __name__ == "__main__":
    print("Model training module loaded successfully!")
