'''Run with:
python3 -m venv env
.\\env\\Scripts\\activate
pip install -r requirements.txt
python PCA.py
'''
'''Essential idea: 
PCA generates a particular set of coordinate
axes (usually in fewer dimensions than the original data) that
capture the maximum variability in the data; furthermore,
these new coordinate axes are orthogonal (which is to say they
are uncorrelated).'''
''' PCA Steps:
1. Standardize the Data: Center the data by subtracting the mean.
2. Measure Feature Relationships: Compute the covariance matrix
3. Find Principal Components: Compute Eigenvalues and Eigenvectors
4. Sort Principal Components: Find direction with max variance and its orthogonal direction
5. Project Data: Transform the original data into the new basis using top-k eigenvectors.
'''

''' from ucimlrepo:
Target: The problem is formulated as a three category classification task 
(dropout, enrolled, and graduate) at the end of the normal duration of the course'''


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 

#from ucimlrepo:
# fetch dataset 
predict_students = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students.data.features 
y = predict_students.data.targets 
y_values = pd.factorize(y.iloc[:,0].values)[0]

# metadata 
#print(predict_students.metadata)  
# variable information 
#print(predict_students.variables)

print('Inputs Dataframe shape   :', X.shape)

#1. Standardize the Data: Center the data by subtracting the mean.
X_mean = np.mean(X, axis=0)  # compute mean for each feature
X_std = np.std(X, axis=0)    # compute standard deviation for each feature
X_standardized = (X - X_mean) / X_std  # standardization

#2. Measure Feature Relationships: Compute the covariance matrix
cov_matrx = np.cov(X_standardized.T) # using transpose to compute columns(features)
sns.heatmap(cov_matrx)
plt.title("Standardized Feature Covariance Matrix")
plt.show()

#3. Find Principal Components: Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrx) # praise be numpy 
print('Eigenvalues:\n', eigenvalues)
print('Eigenvalues Shape:', eigenvalues.shape)
print('Eigenvector Shape:', eigenvectors.shape)

#4. Sort Principal Components: Find direction with max variance and its orthogonal direction
sorted_indices = np.argsort(eigenvalues)[::-1] #sort eigenvalues in descending order to find max eigenvalue
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

#4. Sort Principal Components: Find direction with max variance and its orthogonal direction
explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Principal Components')
plt.grid(True)
plt.show()

# Find minimum k for 95% variance
d = np.argmax(explained_variance>= 0.5) + 1  
print(f"Number of principal components (d*): {d}")

#5. Project Data: Transform the original data into the new basis using top-k eigenvectors.
W = eigenvectors[:, :d]  # select first k eigenvectors
X_pca = X_standardized @ W  # oroject data onto new subspace

# Plot the PCA components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca.iloc[:, 0], X_pca.iloc[:, 1], c=y_values, cmap="plasma", alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection (Manual Algorithm)')
plt.colorbar(label='Target Labels, 0=Dropout, 1=Graduate, 2=Enrolled')
plt.legend()
plt.show()

# Final Step: Print Shape of Reduced Data
print(f"Shape of reduced dataset: {X_pca.shape}")

#Alternative, scikit-learn has its own PCA implementation:

# Standardize the data with sklearn StandardScalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply sklearn's PCA
pca = PCA(n_components=2)  # reduce to 2 components
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe 
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Plot the PCA components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_values, cmap="plasma", alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection (Scikit-Learn Module)')
plt.colorbar(label='Target Labels: 0=Dropout, 1=Graduate, 2=Enrolled')
plt.legend()
plt.show()

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
