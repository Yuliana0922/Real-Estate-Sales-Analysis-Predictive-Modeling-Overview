Real Estate Sales Analysis & Predictive Modeling
Overview

This project involves analyzing and predicting real estate sales prices using historical data from 2001 to 2022. The goal is to engineer features, clean the data, and use machine learning models to predict future sales prices based on factors like assessed value, property type, residential type, and more. The project compares various regression models to find the best predictor of real estate prices.
Table of Contents

    Overview
    Project Structure
    Installation
    Data
    Feature Engineering
    Modeling
    Evaluation
    Results
    Conclusion
    Future Work

Project Structure

bash

Real_Estate_Sales_Project/
│
├── data/
│   └── Real_Estate_Sales_2001-2022_GL.csv
│
├── .venv/                # Virtual environment folder
├── README.md             # This file
├── eda_real_estate.ipynb # Jupyter notebook for EDA
├── model_real_estate.py  # Script for data preprocessing, model training, and evaluation

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/real-estate-sales-analysis.git
cd real-estate-sales-analysis

Set up the virtual environment:

    Create a virtual environment:

    bash

python -m venv .venv

Activate the virtual environment:

    Windows:

    bash

.venv\Scripts\activate

macOS/Linux:

bash

        source .venv/bin/activate

Install the required dependencies:

bash

    pip install -r requirements.txt

Data

The dataset contains real estate sales data from 2001 to 2022. Each row represents a property sale with details such as:

    Assessed Value: The value assessed by local authorities.
    Sale Amount: The final sale price of the property.
    Sales Ratio: The ratio between sale price and assessed value.
    Property Type: The type of property (e.g., Residential, Commercial).
    Residential Type: The specific residential type (e.g., Single Family, Condo).

The dataset contains missing values and extreme outliers that were handled through cleaning and preprocessing steps.
Feature Engineering

New features were engineered to improve model accuracy:

    Season Sold: Categorizes the sale into Winter, Spring, Summer, or Fall based on the sale date.
    One-hot encoding: Categorical variables like property type and residential type were converted into numerical format.

Modeling

Several machine learning models were trained to predict real estate prices, including:

    Linear Regression: A basic model that serves as a baseline.
    Random Forest Regressor: A more advanced tree-based ensemble method.
    Gradient Boosting Regressor: A boosting algorithm that builds models sequentially to reduce errors.

Code Example

Here’s how we trained multiple models in the project:

python

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE = {rmse}, R² = {r2}")

Evaluation

Each model was evaluated using:

    Root Mean Squared Error (RMSE): A measure of how well the model predicts the sales prices, with lower values being better.
    R² Score: Indicates how well the independent variables explain the variance in the sale price, with 1.0 being the perfect score.

For better reliability, we also used cross-validation to ensure model stability.
Results

    Linear Regression: RMSE = 589998.37, R² = 0.0989
    Random Forest: RMSE = 49151.88, R² = 0.9937
    Gradient Boosting: RMSE = 79876.84, R² = 0.9834

The Random Forest model provided the best performance, with a very high R² score and low RMSE, indicating that it is a reliable predictor for real estate prices.
Conclusion

    Data preprocessing and feature engineering significantly improved the model's predictive ability.
    Random Forest outperformed other models in this dataset, making it the best choice for predicting future real estate sales.

Future Work

    Further investigation of other advanced models like XGBoost or LightGBM.
    Implementing geospatial features such as proximity to schools or commercial centers.
    Analyzing the effect of economic indicators (interest rates, inflation) on property sales.
