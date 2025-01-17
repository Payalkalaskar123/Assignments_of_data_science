# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:13:46 2024

@author: SAINATH
"""
'''Problem Statement: -
Perform hierarchical and K-means clustering on the dataset.
 After that, perform PCA on the dataset and extract the
 first 3 principal components and make a new dataset 
 with these 3 principal components as the columns. Now,
 on this new dataset, perform hierarchical and K-means
 clustering. Compare the results of clustering on the 
 original dataset and clustering on the principal 
 components dataset (use the scree plot technique to 
 obtain the optimum number of clusters in K-means 
 clustering and check if you’re getting similar results
 with and without PCA).
'''
from numpy.linalg import eig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df_seg=pd.read_csv("C:/9-pca/alcohol.csv")
df_seg.head()
#we plot a data
#1st we perform kmeans clustering on dataset
plt.figure(figsize=(11,8))
plt.scatter(df_seg[['Alcohol','Malic','Ash','Alcalinity','Magnesium','Phenols','Flavanoids']],df_seg[['Alcohol','Nonflavanoids','Proanthocyanins','Color','Hue','Dilution','Proline']])
plt.xlabel("Ash")
plt.ylabel('Dilution')
plt.title('scatter Plot')
plt.show()

scaler=StandardScaler()
seg_std=scaler.fit_transform(df_seg)
#then we apply pca 
pca=PCA()
pca.fit(seg_std)
pca.explained_variance_ratio_
plt.figure(figsize=(11,8))
plt.plot(pca.explained_variance_ratio_.cumsum(),marker='*')
plt.xlabel("")
plt.ylabel('Dilution')
plt.title('scatter Plot')
plt.show()

#makind data mean centric
Meanbycolumn=np.mean(df_seg.T,axis=1)
print(Meanbycolumn)
Scaled_Data=df_seg-Meanbycolumn
df_seg.T#we need to operate on feature there is 2 feature and scaling is required for to make a data unique
Scaled_Data
#find the covariance matrics od above scaled data
Cov_mat=np.cov(Scaled_Data.T)
Cov_mat
#find corresponding eigen value and eigen vectir of above covarince matrix
Eval,Evec=eig(Cov_mat)
print(Eval)
print(Evec)
#get original data projected to principal component as new axis
Projected_data=Evec.T.dot(Scaled_Data.T)
print(Projected_data.T)
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit_transform(df_seg)

#we choose 3 components 
pca=PCA(n_components=3)
#fit the model the our data with the selected number of component,in our case three
pca.fit_transform(seg_std)
scores_pca=pca.transform(seg_std)
#we create new data frame with the original features and add the pca scores and assigned cluster
df_segm_pca_kmeans=pd.concat([df_seg.reset_index(drop=True),pd.DataFrame(scores_pca)],axis=1)
#s we take three column as our component
df_segm_pca_kmeans.columns.values[-3:]=['component1','component2','component3']
df_segm_pca_kmeans
plt.figure(figsize=(11,9))
plt.plot(df_segm_pca_kmeans[['component1','component2']],df_segm_pca_kmeans['component3'])
plt.xlabel('component 1,2')
plt.ylabel("component2,3")
plt.title("3 component clustering plot")
plt.show()
plt.plot(pca.explained_variance_ratio_.cumsum(),marker='*')
plt.xlabel('component 1,2')
plt.ylabel("component2,3")
plt.title("3 component clustering plot")
plt.show()




'''A pharmaceuticals manufacturing company is conducting 
a study on a new medicine to treat heart diseases. The 
company has gathered data from its secondary sources and
would like you to provide high level analytical insights 
on the data. Its aim is to segregate patients depending 
on their age group and other factors given in the data.
Perform PCA and clustering algorithms on the dataset and
check if the clusters formed before and after PCA are the 
same and provide a brief report on your model. You can 
also explore more ways to improve your model. '''
from numpy.linalg import eig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
file_path ="C:/9-pca/medicine.csv"
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# Drop rows with missing values
df_clean = df.dropna()

# Assuming 'age' is the column for patient age. Modify if necessary.
age_bins = [0, 30, 45, 60, 100]
age_labels = ['<30', '30-45', '45-60', '60+']
df_clean['age_group'] = pd.cut(df_clean['age'], bins=age_bins, labels=age_labels)

# Select features for clustering (Exclude non-numeric columns like 'age_group' and any identifiers)
numeric_features = df_clean.select_dtypes(include=[np.number])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

# Step 3: Apply PCA (Keep enough components to explain ~90% variance)
pca = PCA(n_components=0.9)
pca_data = pca.fit_transform(scaled_data)

# Step 4: Clustering (Before PCA)
kmeans = KMeans(n_clusters=4, random_state=42)  # You can choose the number of clusters based on the elbow method
clusters_before_pca = kmeans.fit_predict(scaled_data)
silhouette_before_pca = silhouette_score(scaled_data, clusters_before_pca)

# Step 5: Clustering (After PCA)
clusters_after_pca = kmeans.fit_predict(pca_data)
silhouette_after_pca = silhouette_score(pca_data, clusters_after_pca)

# Step 6: Compare Results
print("Silhouette Score before PCA:", silhouette_before_pca)
print("Silhouette Score after PCA:", silhouette_after_pca)

# Step 7: Visualization of Clusters
# Plotting the clusters before PCA
plt.figure(figsize=(8,6))
sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=clusters_before_pca, palette='Set1')
plt.title("Clusters Before PCA")
plt.show()

# Plotting the clusters after PCA
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters_after_pca, palette='Set1')
plt.title("Clusters After PCA")
plt.show()








