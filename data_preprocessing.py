import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def load_and_prepare_data():
    """Load California Housing Dataset"""
    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    
    return df

def handle_missing_values(df):
    """Handle missing values"""
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Fill missing values if any
    df = df.fillna(df.median())
    return df

def split_data(df):
    """Split data into train and test sets"""
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save processed data
    df.to_csv('housing_data.csv', index=False)
    print("\n✅ Data preprocessing complete!")
