

# 🏠 House Price Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)

A machine learning-based house price prediction system using XGBoost algorithm on California Housing dataset. This project includes data preprocessing, model training, evaluation, and an interactive web application built with Streamlit.

## 📊 Project Overview

This project predicts house prices based on various features such as location, median income, house age, and more. The model achieves an **R² score of ~0.84**, making it highly accurate for real estate price estimation.

### Key Features

- 🎯 **High Accuracy**: R² score of 0.84 (explains 84% of price variation)
- ⚡ **Fast Predictions**: Real-time price estimation in milliseconds
- 🌐 **Interactive Dashboard**: User-friendly Streamlit web interface
- 📈 **Data Visualization**: Comprehensive EDA and model evaluation plots
- 🔄 **Complete Pipeline**: End-to-end ML workflow from data to deployment

## 🖼️ Screenshots

### Dashboard
![Dashboard](outputs/dashboard_screenshot.png)

### Model Performance
![Evaluation](outputs/model_evaluation.png)

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn, XGBoost |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Web Framework** | Streamlit |
| **Model Persistence** | Pickle |

## 📁 Project Structure

```
house-price-prediction/
│
├── data_preprocessing.py      # Data loading and cleaning
├── data_visualization.py      # Exploratory Data Analysis
├── model_training.py          # Model training and comparison
├── model_evaluation.py        # Performance evaluation
├── app.py                     # Streamlit web application
├── run_pipeline.py            # Complete pipeline execution
│
├── housing_data.csv           # Processed dataset
├── house_price_model.pkl      # Trained XGBoost model
├── feature_names.pkl          # Feature names for prediction
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create virtual environment** (Windows)
   ```
   python -m venv env
   .\env\Scripts\Activate.ps1
   ```

   *For Linux/Mac:*
   ```
   python -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Run Complete Pipeline

```
python run_pipeline.py
```

This will execute:
- Data preprocessing
- Data visualization
- Model training
- Model evaluation

#### Option 2: Run Individual Steps

```
# Step 1: Preprocess data
python data_preprocessing.py

# Step 2: Visualize data
python data_visualization.py

# Step 3: Train models
python model_training.py

# Step 4: Evaluate model
python model_evaluation.py
```

#### Launch Web Application

```
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📊 Dataset

**Source**: California Housing Dataset (sklearn)  
**Origin**: 1990 U.S. Census  
**Samples**: 20,640 houses  
**Features**: 8 input features + 1 target variable

### Features Description

| Feature | Description | Type |
|---------|------------|------|
| **MedInc** | Median income in block group | Continuous |
| **HouseAge** | Median house age in block group | Continuous |
| **AveRooms** | Average number of rooms per household | Continuous |
| **AveBedrms** | Average number of bedrooms per household | Continuous |
| **Population** | Block group population | Continuous |
| **AveOccup** | Average number of household members | Continuous |
| **Latitude** | Block group latitude | Continuous |
| **Longitude** | Block group longitude | Continuous |
| **Price** | Median house value (target) | Continuous |

## 🧠 Model Performance

### Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.84 | Explains 84% of variance |
| **RMSE** | 0.50 | Average error of $50k |
| **MAE** | 0.32 | Mean absolute error of $32k |

### Model Comparison

| Algorithm | R² Score | Training Time |
|-----------|----------|---------------|
| Linear Regression | 0.62 | 0.1s |
| Ridge Regression | 0.63 | 0.1s |
| Random Forest | 0.79 | 2.5s |
| **XGBoost** | **0.84** | **3.2s** |

**Winner**: XGBoost (Best accuracy-speed tradeoff)

## 🎨 Web Application Features

### Input Parameters
- Adjustable sliders for all 8 features
- Real-time prediction updates
- Input validation and range constraints

### Output
- Predicted house price in USD
- Price per room calculation
- Price per bedroom calculation
- Model performance metrics display

### Visualizations
- Actual vs Predicted scatter plot
- Residual distribution plot
- Feature importance chart



## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- California Housing Dataset from [scikit-learn](https://scikit-learn.org/)
- XGBoost Library by [DMLC](https://github.com/dmlc/xgboost)
- Streamlit framework for rapid app development
- Machine Learning community for inspiration




⭐ **Star this repository** if you found it helpful!

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
```



