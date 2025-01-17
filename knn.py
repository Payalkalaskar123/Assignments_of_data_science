# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:48:10 2024

@author: Payal
"""


'''

### **1. Business Problem**

#### **1.1 What is the business objective?**  
Clearly define the goal the business seeks to achieve. For example:  
- **Goal:** Reduce customer churn, optimize marketing campaigns, predict sales, etc.  
- Explain the expected outcome of the analysis or model in terms of business value.

#### **1.2 Are there any constraints?**  
Identify limitations such as:  
- **Data constraints:** Missing or incomplete data, limited historical data, etc.  
- **Time constraints:** Deadlines for delivery.  
- **Model constraints:** Preference for interpretable models, limited computational resources.  
- **Regulatory constraints:** GDPR compliance or domain-specific regulations.

---

### **2. Data Dictionary**

#### **2.1 Feature Description Table**  
For each feature in the dataset, create a structured table as follows:

| **Feature Name** | **Data Type** | **Description**                 | **Relevance to Model** | **Reason (if not relevant)**              |
|-------------------|---------------|---------------------------------|-------------------------|-------------------------------------------|
| CustomerID        | Categorical   | Unique identifier for customers | Not Relevant           | No predictive value; only for record-keeping. |
| Age               | Numerical     | Age of the customer             | Relevant               | Useful for segmenting customer behavior. |
| PurchaseHistory   | Categorical   | Record of past purchases        | Relevant               | May indicate customer preferences.       |
| City              | Categorical   | Location of customer            | May be relevant        | Could correlate with purchasing behavior. |

---

### **3. Data Preprocessing**

#### **3.1 Data Cleaning and Feature Engineering**  
- **Handling Missing Data:** Impute or remove missing values depending on their significance.  
- **Feature Engineering:** Create new features if necessary, such as `Recency`, `Frequency`, or `Monetary value` (RFM analysis).  
- **Encoding:** Convert categorical data to numerical using techniques like one-hot encoding or label encoding.  
- **Scaling:** Apply normalization or standardization to ensure features contribute equally to distance metrics (essential for KNN).  

---

### **4. Exploratory Data Analysis (EDA)**

#### **4.1 Summary Statistics**  
- Calculate mean, median, mode, standard deviation, and other descriptive statistics for all numerical features.  

#### **4.2 Univariate Analysis**  
- Use visualizations (histograms, boxplots) to analyze the distribution of individual features.

#### **4.3 Bivariate Analysis**  
- Explore relationships between features using correlation heatmaps, scatterplots, or pair plots.  
- Identify patterns or outliers that may affect the model.

---

### **5. Model Building**

#### **5.1 Scaled Data**  
- Experiment with different scaling methods (e.g., MinMaxScaler, StandardScaler) to prepare the data.  

#### **5.2 KNN with Cross-Validation**  
- Use cross-validation to determine the optimal value of `K` for the KNN model.  
- Plot the error rate versus `K` to identify the ideal number of neighbors.

#### **5.3 Train-Test Split and Model Evaluation**  
- Split the data into training and test sets.  
- Evaluate the model using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.  
- Document and compare results across multiple iterations.

#### **5.4 Model Output Documentation**  
- Summarize the model’s performance and insights.  
- Provide an explanation of how predictions align with business goals.

---

### **6. Benefits/Impact of the Solution**

Explain how the solution addresses the business objective:  
- **For the client:** Improved decision-making, cost savings, or revenue growth.  
- **Example:** A churn prediction model helps retain high-value customers by targeting them with personalized offers, reducing churn rate by 10%.  
- **Quantify impact:** Use metrics (e.g., projected increase in revenue or decrease in costs) to showcase value.  

---

This structure ensures a systematic approach to solving the business problem, with clear deliverables at each step. Let me know if you'd like further clarification or assistance!
'''
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = "C:/Users/Rohit/Downloads/glass.csv"  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Shape:", data.shape)
print(data.info())
print(data.describe())

# Step 1: Data Preprocessing
# Separating features and target variable
X = data.iloc[:, :-1]  # Assuming last column is the target
y = data.iloc[:, -1]   # Assuming last column is the target

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Exploratory Data Analysis (optional visualizations can be added)
print("\nCorrelation matrix:")
print(data.corr())

# Step 3: Model Building and Evaluation
# Determine the optimal value of K using cross-validation
error_rate = []
for k in range(1, 21):  # Test K values from 1 to 20
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    error_rate.append(1 - np.mean(scores))

# Plotting error rate vs. K
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# Train the KNN model with the optimal K value
optimal_k = error_rate.index(min(error_rate)) + 1
print("Optimal K:", optimal_k)

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train_scaled, y_train)

# Predictions and Evaluations
y_pred = knn.predict(X_test_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

#Problem Statement 2
# Import necessary libraries
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
# Assuming data is loaded from a CSV file (replace with actual file path if needed)
# data = pd.read_csv("C:/Users/Rohit/Downloads/Zoo.csv")

# Example dataset (you would replace this with your actual data)
data = pd.DataFrame({
    'Size': [200, 600, 50, 300, 700],  # Size in kg (e.g., elephant, tiger, deer)
    'Diet': [1, 0, 2, 0, 1],  # 1: Herbivore, 0: Carnivore, 2: Omnivore
    'Habitat': [0, 1, 0, 2, 1],  # 0: Forest, 1: Grassland, 2: Desert
    'Speed': [25, 15, 40, 10, 45],  # Speed in km/h
    'Lifespan': [15, 60, 10, 20, 30],  # Lifespan in years
    'Endangerment': [1, 0, 1, 0, 2],  # 0: Least Concern, 1: Endangered, 2: Critically Endangered
    'Label': ['Elephant', 'Tiger', 'Deer', 'Lion', 'Cheetah']  # Target (species)
})

# Step 2: Separate features (X) and target labels (y)
X = data.drop(columns=["Label"])  # Features (all columns except 'Label')
y = data["Label"]  # Target (species)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform the training set
X_test = scaler.transform(X_test)  # Only transform the test set

# Step 5: Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # Start with k=3
knn.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = knn.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Tune K value (Optional)
# You can try different values of K and plot accuracy to select the best K
k_values = list(range(1, 21))  # Trying K values from 1 to 20
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

# Plot accuracy for different K values
plt.plot(k_values, accuracies)
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. K for KNN")
plt.show()

