# House Price Prediction Using Machine Learning

A complete regression project including Exploratory Data Analysis (EDA), preprocessing, feature engineering, multiple machine learning models, and model performance comparison.

## Project Overview

This project predicts house prices using various machine learning regression algorithms.  
It follows a full data science lifecycle that includes:

- Exploratory Data Analysis  
- Data cleaning and preprocessing  
- Feature engineering  
- Model training  
- Model evaluation and comparison  
- Feature importance analysis  

The objective is to identify the best-performing regression model and understand the key factors influencing house prices.

## Dataset Credit

The Housing dataset used in this project comes from the Kaggle “Housing Price Prediction Dataset”:
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

### Numeric Features
- area  
- bedrooms  
- bathrooms  
- stories  
- parking  

### Binary Features (yes/no)
- mainroad  
- guestroom  
- basement  
- hotwaterheating  
- airconditioning  
- prefarea  

### Categorical Feature
- furnishingstatus (furnished, semi-furnished, unfurnished)

### Target Variable
- price  

The dataset contains no missing values and no duplicate rows.

## Exploratory Data Analysis (EDA)

The EDA phase included:

### Distribution Analysis  
- Price distribution  
- Area, bedrooms, bathrooms, stories, and parking distributions  

### Categorical Analysis  
- Countplots for all yes/no features  
- Distribution of furnishingstatus  

### Relationship Analysis  
- Scatter plots of numeric variables vs. price  
- Boxplots comparing categorical variables to price  

### Correlation Analysis  
- Correlation heatmap of numeric features  
- Pairplot of numeric features  

### Key Observations
- House area has a strong positive correlation with price.  
- More bathrooms or more stories generally increase price.  
- Houses located in preferred areas tend to be more expensive.  
- Furnished homes typically have higher prices compared to unfurnished ones.

## Data Preprocessing

The following preprocessing steps were applied:

### Binary Encoding
All yes/no columns were mapped to 1 and 0.

### One-Hot Encoding
The furnishingstatus column was encoded using one-hot encoding.

### Boolean Cleanup
Any remaining boolean values were converted to integers.

### Feature Engineering
Two new features were created:
- price_per_area = price / area  
- total_rooms = bedrooms + bathrooms  

### Train-Test Split
The dataset was split as follows:
- 80% training data  
- 20% testing data  

### Feature Scaling
StandardScaler was applied to all numeric features for linear models.  
Tree-based models were trained on unscaled data.

## Machine Learning Models and Results

The following regression models were trained:

| Model                | RMSE        | R² Score |
|---------------------|-------------|----------|
| Linear Regression   | 766300.914  | 0.884    |
| Ridge Regression    | 769482.459  | 0.883    |
| Lasso Regression    | 766290.325  | 0.884    |
| Decision Tree       | 659273.947  | 0.914    |
| Random Forest       | 573974.277  | 0.935    |

### Best Model: Random Forest

Random Forest achieved:
- Lowest RMSE  
- Highest R² score  
- Strong generalization to unseen data  

Reasons for strong performance:
- Ability to model non-linear relationships  
- Robustness against noise  
- Automatic handling of feature interactions  

## Feature Importance Analysis

Using the Random Forest model, the most influential features were:

1. area  
2. bathrooms  
3. stories  
4. prefarea  
5. furnishingstatus  
6. price_per_area  
7. total_rooms  

These insights align with real-world expectations about factors affecting house prices.

## Results Summary

- Linear models provided solid baselines but underperformed on non-linear relationships.  
- Ridge and Lasso improved regularization but produced similar performance to linear regression.  
- Decision Tree significantly outperformed linear models.  
- Random Forest was the strongest model overall, achieving the best RMSE and R² scores.  
- The final model explains approximately 93.5% of the variation in house prices.

## Future Work

Possible enhancements include:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV  
- Testing more advanced models such as XGBoost, LightGBM, or CatBoost  
- Incorporating geographical location features  
- Adding cross-validation for improved reliability  
- Deploying the model using Flask, FastAPI, or Streamlit  

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  

---

## Author

**Noor Hazem Ibrahim Seif**  
Computer Science Student – American University of Sharjah  
LinkedIn: https://www.linkedin.com/in/noor-seif-39babb328/
