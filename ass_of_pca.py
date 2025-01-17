# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:31:46 2024

@author: SAINATH
"""

'''4. Apply both PCA and SVD for dimensionality reduction on the Breast 
Cancer dataset, and compare the results in terms of variance explained 
and reconstruction accuracy.
Goals:
 Apply PCA to the standardized dataset and reduce it to 5 components.
 Perform SVD on the same dataset, also reducing it to 5 components.
 Compare the explained variance for PCA and the reconstruction accuracy for both methods.
 Calculate and report the reconstruction error (mean squared error) for both methods.'''
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data

# Standardize the dataset
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 1. Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_std)

# Explained variance for PCA
explained_variance_pca = np.sum(pca.explained_variance_ratio_)

# Reconstruct the data from PCA components
X_pca_reconstructed = pca.inverse_transform(X_pca)

# Calculate reconstruction error (MSE) for PCA
mse_pca = mean_squared_error(X_std, X_pca_reconstructed)

# 2. Apply SVD
U, Sigma, VT = randomized_svd(X_std, n_components=5, random_state=42)

# Reconstruct the data using SVD components
X_svd_reconstructed = np.dot(U, np.dot(np.diag(Sigma), VT))

# Calculate reconstruction error (MSE) for SVD
mse_svd = mean_squared_error(X_std, X_svd_reconstructed)

explained_variance_pca, mse_pca, mse_svd

'''2. Perform PCA on the iris dataset to understand how much variance is 
explained by each principal component. Additionally, you must determine 
how many principal components are required to capture at least 95% of 
the total variance in the data.
Goals:
 Apply PCA to the standardized Iris dataset.
 Calculate and plot the cumulative explained variance for each principal component.
 Identify the minimum number of components needed to explain 95% of the variance.
 Visualize this with a plot that shows the cumulative explained variance.'''

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate the cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine the number of components needed to explain 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

# Plot the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance, marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

n_components_95, cumulative_variance

'''3. Perform Singular Value Decomposition (SVD) on a randomly generated 
matrix and verify that the original matrix can be reconstructed using the 
product of the decomposed matrices.
Goals:
 Generate a random matrix of size 5x5.
 Perform SVD on this matrix to obtain the U, Σ (singular values), and Vᵀ matrices.
 Reconstruct the original matrix using the decomposed matrices.
 Compare the original and reconstructed matrices and compute the difference.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
marks=[[[[[[1,2],[3,4],[5,6],[7,8],[9,8]]]]]]
marks
marks=pd.DataFrame(marks,columns=['A','B','C','D'',E'])
marks







