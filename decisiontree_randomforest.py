# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:29:51 2024

@author: Payal

"""
'''

#### **1. Business Problem**
1.1 **What is the business objective?**
   - Clearly define the problem the business aims to solve. For example:
     - *Predicting customer churn* for a telecom company.
     - *Identifying fraudulent transactions* for a financial institution.
     - *Forecasting sales* for a retail business.
   - The objective should align with business goals, such as improving customer retention or optimizing costs.

1.2 **Are there any constraints?**
   - **Data-related constraints** (e.g., limited data availability or missing values).
   - **Computational constraints** (e.g., time required to train a model or deploy it).
   - **Interpretability requirements** (e.g., business prefers a simple, interpretable model like a Decision Tree).
   - **Regulatory constraints** (e.g., data privacy laws like GDPR).

---

#### **2. Data Dictionary**
**Create a table to define each feature in the dataset.**

| **Feature Name** | **Data Type** | **Description**                      | **Relevance**              | **Reason if Irrelevant**       |
|-------------------|---------------|--------------------------------------|----------------------------|---------------------------------|
| `age`            | Numeric       | Age of the customer                 | Relevant                   | N/A                             |
| `income`         | Numeric       | Monthly income of the customer      | Relevant                   | N/A                             |
| `gender`         | Categorical   | Gender of the customer              | Relevant                   | N/A                             |
| `purchase_freq`  | Numeric       | Number of purchases in a year       | Relevant                   | N/A                             |
| `customer_id`    | Identifier    | Unique identifier for each customer | Irrelevant for modeling    | Used only for tracking records |
| `region`         | Categorical   | Region where the customer resides   | Relevant (may impact sales)| N/A                             |

---

#### **3. Data Pre-Processing**
3.1 **Steps for Data Cleaning and Feature Engineering**
   - **Handle Missing Values**:
     - Impute numeric features with mean/median.
     - Impute categorical features with mode.
   - **Remove Duplicates** to ensure data integrity.
   - **Outlier Treatment**:
     - Use statistical methods like IQR or z-scores to identify and address outliers.
   - **Feature Scaling**:
     - Apply standardization or normalization if required for models sensitive to feature scales (e.g., Random Forest doesn't need scaling, but logistic regression does).
   - **Encode Categorical Variables**:
     - Use one-hot encoding or label encoding.
   - **Feature Selection**:
     - Drop irrelevant features like `customer_id`.

---

#### **4. Exploratory Data Analysis (EDA)**
4.1 **Summary**
   - Summarize the dataset with descriptive statistics (mean, median, standard deviation).
   - Identify key insights like the percentage of missing values, class imbalance, etc.

4.2 **Univariate Analysis**
   - Analyze each feature independently:
     - Distribution plots (e.g., histograms, box plots for numeric variables).
     - Bar charts for categorical variables.

4.3 **Bivariate Analysis**
   - Analyze relationships between features:
     - Scatter plots for numeric-numeric relationships.
     - Box plots for numeric-categorical relationships.
     - Correlation matrix to understand the relationship between numeric features.

---

#### **5. Model Building**
5.1 **Prepare Data for Model Training**
   - Split the data into training and testing datasets (e.g., 80-20 split).
   - Apply scaling if necessary.

5.2 **Build Decision Tree and Random Forest Models**
   - **Decision Tree**:
     - Train a Decision Tree using hyperparameters like maximum depth and minimum samples split.
   - **Random Forest**:
     - Use multiple Decision Trees with bagging to improve accuracy and reduce overfitting.

5.3 **Evaluation Metrics**
   - Train, test, and cross-validate the models.
   - Evaluate using metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC-AUC
   - Compare the results to determine the better model for the problem.

5.4 **Model Documentation**
   - Summarize:
     - Key hyperparameters used.
     - Model performance metrics.
     - Interpretation of results (e.g., feature importance in Random Forest).

---

#### **6. Business Benefits and Impact**
   - **Improved Decision-Making**: Accurate predictions lead to actionable insights.
   - **Cost Efficiency**: Automating predictions reduces manual effort and errors.
   - **Customer Retention**: Targeted actions based on predictions (e.g., identifying high churn risk).
   - **Scalability**: Deploying a Random Forest model allows the business to handle complex datasets effectively.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("C:/Users/Rohit/Downloads/Company_Data.csv")

# Step 1: Convert Sales into a categorical variable
median_sales = data['Sales'].median()
data['SalesCategory'] = data['Sales'].apply(lambda x: 'High' if x > median_sales else 'Low')

# Drop the original Sales column
data = data.drop(columns=['Sales'])

# Step 2: Encode categorical variables
data = pd.get_dummies(data, columns=['ShelveLoc'], drop_first=True)  # One-hot encode ShelveLoc
data['Urban'] = data['Urban'].map({'Yes': 1, 'No': 0})              # Binary encode Urban
data['US'] = data['US'].map({'Yes': 1, 'No': 0})                    # Binary encode US
data['SalesCategory'] = data['SalesCategory'].map({'Low': 0, 'High': 1})  # Encode target variable

# Step 3: Split the data into features and target
X = data.drop(columns=['SalesCategory'])
y = data['SalesCategory']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate the Decision Tree
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Results:")
print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))

# Step 5: Random Forest Model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluate the Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Step 6: Cross-validation for Random Forest
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Random Forest Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Step 7: Feature Importance (Random Forest)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importance in Random Forest')
plt.show()

#Problem statement 2
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the diabetes dataset
data = pd.read_csv("C:/Users/Rohit/Downloads/Diabetes.csv")

# Separate features and target variable
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training  and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

# Build a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_predictions))

# Build a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
dt_predictions = dt_model.predict(X_test)

# Evaluate the Decision Tree model
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)
print(classification_report(y_test, dt_predictions))

#Problem Statement 3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv("C:/Users/Rohit/Downloads/Fraud_check.csv")  # Replace with your actual file path

# Discretize the 'taxable_income' column
data['risk_category'] = pd.cut(data['taxable_income'], bins=[-float('inf'), 30000, float('inf')], labels=['Risky', 'Good'])

# Separate features and target variable
X = data.drop('risk_category', axis=1)
y = data['risk_category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree Model
print("Decision Tree Model Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest Model
print("\nRandom Forest Model Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

#Problem Statement 4
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:/Users/Rohit/Downloads/HR_DT.csv")  # Replace with the actual file path

# Step 1: Data Cleaning
# Rename columns for easier handling
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
data.rename(columns={
    'Position of the employee': 'Position',
    'no of Years of Experience of employee': 'Experience',
    'monthly income of employee': 'MonthlyIncome'
}, inplace=True)

# Encode the categorical variable (Position)
le = LabelEncoder()
data['Position'] = le.fit_transform(data['Position'])

# Step 2: Define Features and Target
X = data.drop(columns=['MonthlyIncome'])  # Features
y = data['MonthlyIncome']                # Target variable

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict using Decision Tree
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree
print("Decision Tree Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_dt):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_dt):.2f}")

# Step 5: Random Forest Model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Predict using Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
print("Random Forest Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_rf):.2f}")

# Step 6: Feature Importance (Random Forest)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importance in Random Forest')
plt.show()
