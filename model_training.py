import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

def train_models():
    """Train multiple models and compare performance"""
    # Load data
    df = pd.read_csv('housing_data.csv')
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    print("Training models...\n")
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'model': model
        }
        
        print(f"{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}\n")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = results[best_model_name]['model']
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   R² Score: {results[best_model_name]['R2']:.4f}")
    
    # Save best model
    with open('house_price_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print("\n✅ Best model saved as 'house_price_model.pkl'")
    
    return best_model, X_test, y_test

if __name__ == "__main__":
    train_models()
