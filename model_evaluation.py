import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

def evaluate_model():
    """Evaluate the trained model"""
    # Load model
    with open('house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    df = pd.read_csv('housing_data.csv')
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create comparison plot
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices')
    plt.legend()
    
    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300)
    plt.show()
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()
        
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())
    
    print("\n✅ Evaluation complete! Plots saved.")

if __name__ == "__main__":
    evaluate_model()
