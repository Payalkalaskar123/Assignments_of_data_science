# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:40:33 2024

@author: Payal
"""

''' Here’s a structured plan to address your topic, *Naïve Bayes*, incorporating the hints provided:

---

## **Naïve Bayes Analysis Plan**

---

### **1. Business Problem**

#### **1.1 What is the Business Objective?**
- Clearly define the goal of the analysis, e.g., predicting customer churn, classifying emails as spam/non-spam, or detecting fraudulent transactions.
- Identify the outcome variable (target) and the predictors (features).

#### **1.2 Are there any Constraints?**
- Constraints may include:
  - Time to deliver results.
  - Computational resources available.
  - Requirements for explainability or interpretability of the model.

---

### **2. Feature Analysis and Data Dictionary**

#### **2.1 Data Dictionary**
Create a table summarizing the dataset's features. Below is an example format:

| **Feature Name** | **Data Type** | **Description** | **Relevance** | **Reason (if not relevant)** |
|-------------------|---------------|------------------|---------------|-----------------------------|
| Age               | Integer       | Customer's age   | Relevant      | -                           |
| Salary            | Float         | Annual income    | Relevant      | -                           |
| Location          | Categorical   | Customer region  | Not relevant  | Lack of correlation         |

For each feature, analyze:
- The data type (categorical, numerical, etc.).
- Its significance in predicting the target variable.
- Any reason for excluding the feature (e.g., multicollinearity or low variance).

---

### **3. Data Pre-processing**

#### **3.1 Data Cleaning**
- Handle missing values (e.g., imputation or removal).
- Remove duplicates and handle outliers.
- Convert categorical variables to numerical using encoding techniques (e.g., one-hot encoding, label encoding).

#### **3.2 Feature Engineering**
- Create new features if necessary.
- Scale features to ensure all are on a similar scale (e.g., Min-Max scaling or standardization).

---

### **4. Exploratory Data Analysis (EDA)**

#### **4.1 Summary**
- Provide an overview of the dataset, including the total number of features, observations, and basic statistical measures (mean, median, etc.).

#### **4.2 Univariate Analysis**
- Examine individual features using:
  - Histograms and density plots for numerical data.
  - Bar charts and frequency tables for categorical data.

#### **4.3 Bivariate Analysis**
- Study relationships between variables using:
  - Correlation matrices for numerical features.
  - Box plots and scatter plots to detect patterns.
  - Heatmaps for visualizing relationships.

---

### **5. Model Building**

#### **5.1 Build the Model**
- Split the dataset into training and test sets.
- Apply scaling techniques to standardize the data.

#### **5.2 Build a Naïve Bayes Model**
- Use libraries like `scikit-learn` to build and fit a Naïve Bayes model:
  - For categorical data: Use the Multinomial Naïve Bayes model.
  - For continuous data: Use the Gaussian Naïve Bayes model.

#### **5.3 Validate the Model**
- Generate a confusion matrix.
- Calculate:
  - **Precision**: \( \text{Precision} = \frac{TP}{TP + FP} \)
  - **Recall**: \( \text{Recall} = \frac{TP}{TP + FN} \)
  - **Accuracy**: \( \text{Accuracy} = \frac{TP + TN}{Total Samples} \)

#### **5.4 Tune the Model**
- Adjust model parameters and preprocess features to improve accuracy.
- Perform cross-validation to evaluate performance consistency.

---

### **6. Benefits/Impact of the Solution**

#### **Business Impact:**
- Quantify the benefits, e.g., improved accuracy leading to better decision-making.
- Highlight how the solution supports the business objectives.
- Examples:
  - Enhanced customer segmentation for targeted marketing.
  - Improved fraud detection, reducing financial losses.
  - Faster, automated classification saving time and resources. 

---

### **Implementation Tools**
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
train_data = pd.read_csv("C:/Users\Rohit/Downloads/SalaryData_Train.csv")
test_data = pd.read_csv("C:/Users/Rohit/Downloads/SalaryData_Test.csv")

# Preprocess the data
# Identify categorical columns (excluding target 'Salary')
categorical_columns = train_data.select_dtypes(include=['object']).columns.drop('Salary')

# Encode categorical variables using LabelEncoder
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    train_data[col] = label_encoders[col].fit_transform(train_data[col])
    test_data[col] = label_encoders[col].transform(test_data[col])

# Encode the target variable 'Salary'
target_encoder = LabelEncoder()
train_data['Salary'] = target_encoder.fit_transform(train_data['Salary'])
test_data['Salary'] = target_encoder.transform(test_data['Salary'])

# Separate features and target
X_train = train_data.drop(columns='Salary')
y_train = train_data['Salary']
X_test = test_data.drop(columns='Salary')
y_test = test_data['Salary']

# Train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on the test data
y_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=target_encoder.classes_)

# Display results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_rep)

#Problem Statement 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("C:/Users/Rohit/Downloads/NB_Car_Ad.csv")

# Drop irrelevant column
data = data.drop(columns=["User ID"])

# Encode the Gender column
label_encoder = LabelEncoder()
data["Gender"] = label_encoder.fit_transform(data["Gender"])

# Extract features and target variable
X = data.drop(columns=["Purchased"])
y = data["Purchased"]

# Normalize numerical features
scaler = StandardScaler()
X[["Age", "EstimatedSalary"]] = scaler.fit_transform(X[["Age", "EstimatedSalary"]])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the Bernoulli Naive Bayes model
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train, y_train)

# Predict on the test set
y_pred = bernoulli_nb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

#Problem Statement 3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
tweets_data = pd.read_csv("C:/Users/Rohit/Downloads/Disaster_tweets_NB.csv")

# Inspect the dataset (optional)
print(tweets_data.head())

# Assume the dataset has 'text' and 'target' columns (update if different)
# 'text' contains the tweet content
# 'target' is the label (1 = real, 0 = fake)

# Extract features and labels
X = tweets_data['text']
y = tweets_data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert text data to feature vectors using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = nb_model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
