# Customer-Churn-Prediction-with-ML-Streamlit

## Overview
This project focuses on predicting customer churn using machine learning. The process includes data analysis, feature engineering, model training, and the development of a Streamlit web app to interact with the trained model.

## Features
- **Data Preprocessing**: Handling missing values, feature selection, encoding categorical variables.
- **Exploratory Data Analysis (EDA)**: Visualization of customer data and churn trends.
- **Feature Engineering**: Scaling and transforming features for better model performance.
- **Machine Learning Models**: Training various models including Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, and Random Forest.
- **Hyperparameter Tuning**: Using GridSearchCV to optimize model performance.
- **Streamlit Web App**: A simple interface to input customer data and get churn predictions.

## Dataset
The dataset contains 10 columns:
- `Customer ID` (Unique identifier)
- `Age` (Customer age)
- `Gender` (Male/Female)
- `Tenure` (Duration of customer subscription in months)
- `Monthly Charges` (Customer's monthly bill)
- `Contract Type` (Monthly, Yearly, etc.)
- `Internet Service` (Type of internet service)
- `Total Charges` (Total bill over tenure)
- `Tech Support` (Availability of tech support)
- `Churn` (Target variable - Yes/No)

## Project Workflow
1. **Data Preprocessing**
   - Load dataset and check for missing values
   - Handle missing values using imputation techniques
   - Encode categorical variables
   - Feature scaling using StandardScaler

2. **Exploratory Data Analysis**
   - Compute summary statistics
   - Visualize data distributions using histograms and pie charts
   - Analyze correlation between features
   - Group-by analysis to explore relationships between features and churn

3. **Model Training**
   - Train multiple machine learning models:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Classifier (SVC)
     - Decision Tree
     - Random Forest
   - Hyperparameter tuning using GridSearchCV
   - Evaluate models using accuracy score

4. **Model Selection & Export**
   - Select the best-performing model (SVC in this case)
   - Export the trained model using joblib

5. **Streamlit Web App**
   - Create an interactive web interface
   - Accept user inputs for prediction
   - Load trained model and scaler
   - Display churn prediction

## Installation & Usage
### Requirements
- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. Run the Jupyter Notebook for data processing and model training:
   ```bash
   jupyter notebook churn_prediction.ipynb
   ```
2. Run the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided URL in a web browser to access the application.

## Project Structure
```
customer-churn-prediction/
│── data/
│   ├── customer_churn.csv  # Dataset
│── models/
│   ├── model.pkl           # Trained model
│   ├── scaler.pkl          # StandardScaler instance
│── notebooks/
│   ├── churn_prediction.ipynb  # Jupyter Notebook for model training
│── app.py                  # Streamlit web app
│── requirements.txt         # Python dependencies
│── README.md                # Project documentation
```

## Results & Insights
- The **Support Vector Classifier (SVC)** model achieved the highest accuracy of **94.5%**.
- Customers with **higher monthly charges** were more likely to churn.
- Customers with **shorter tenure periods** showed a higher churn rate.
- Adding **contract type** and **tech support** as features could improve model performance.

## Future Improvements
- Implement additional data balancing techniques for better generalization.
- Deploy the Streamlit app on a cloud platform for public access.
- Experiment with deep learning models for better accuracy.

## Author
Project inspired by [YouTube Tutorial](https://www.youtube.com/watch/XRnQUQmS2_s). Feel free to reach out for any questions!

## License
This project is open-source and available under the MIT License.

