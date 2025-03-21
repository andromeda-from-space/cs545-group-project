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
import preprocessing

def pca_manual(variance):

    # fetch dataset 
    X, y, _, _ = preprocessing.main()
    target_column = np.argmax(y, axis=1)

    # Map numeric labels to class names
    class_names = ['dropout', 'enrolled', 'graduate']
    target_column_names = [class_names[label] for label in target_column]

    #1. Standardize the Data: Center the data by subtracting the mean.
    X_mean = np.mean(X, axis=0)  # compute mean for each feature
    X_std = np.std(X, axis=0)    # compute standard deviation for each feature
    X_standardized = (X - X_mean) / X_std  # standardization

    #2. Measure Feature Relationships: Compute the covariance matrix
    cov_matrx = np.cov(X_standardized.T) # using transpose to compute columns(features)
    sns.heatmap(cov_matrx)
    plt.title("Standardized Feature Covariance Matrix")
    plt.savefig('Covariance Matrix')
    plt.close()

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
    plt.savefig('Explained Variance')
    plt.close()

    # Find minimum k for 95% variance
    d = np.argmax(explained_variance>= variance) + 1  
    print(f"Number of principal components (d*): {d}")

    #5. Project Data: Transform the original data into the new basis using top-k eigenvectors.
    W = eigenvectors[:, :d]  # select first k eigenvectors
    pca = X_standardized @ W  # project data onto new subspace
    print(f"Applied PCA. Reduced dimensions from {X[1]} to {pca.shape[1]}")

    # Convert to dataframe 
    column_names = [f'PC{i+1}' for i in range(d)]
    pca_df = pd.DataFrame(pca, columns=column_names)
    pca_df['Outcome'] = target_column_names  # Use class names instead of numeric labels
    
    # Generate graph of tsne output
    display_results(pca_df, d, variance)

    # Final Step: Print Shape of Reduced Data
    print(f"Shape of reduced dataset: {pca.shape}")

#Alternative, scikit-learn has its own PCA implementation:
def pca_sklearn(n_components):

    # Get preprocessed data
    train_data, train_label, _, _ = preprocessing.main()
    target_column = np.argmax(train_label, axis=1)

    # Map numeric labels to class names
    class_names = ['dropout', 'enrolled', 'graduate']
    target_column_names = [class_names[label] for label in target_column]
    '''
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        train_data_pca = pca.fit_transform(train_data)
        test_data_pca = pca.transform(test_data)
        
        print(f"Applied PCA. Reduced dimensions from {train_data.shape[1]} to {train_data_pca.shape[1]}")
        return train_data_pca, train_label, test_data_pca, test_label
    '''

    # Apply sklearn's PCA
    pca = PCA(n_components)  # reduce to 2 components
    train_pca = pca.fit_transform(train_data)
    print(f"Applied PCA. Reduced dimensions from {train_data.shape[1]} to {train_pca.shape[1]}")

    # Convert to dataframe
    column_names = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(train_pca, columns=column_names)
    pca_df['Outcome'] = target_column_names  # Use class names instead of numeric labels
    
    # Generate graph of tsne output
    display_results(pca_df, n_components, 'N/A')

    # Print explained variance ratio
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

def display_results(pca_df, num_components, variance):

    title = f'PCA Visualization\nNumber of Principle Components={num_components}, Variance={variance}' 
    filename = f'PCA num_components={num_components}'

    custom_palette = {
        'dropout': 'red',
        'graduate': 'green',
        'enrolled': 'blue'
    }

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue=pca_df['Outcome'],  # Color-code by class names
        palette=custom_palette,
        data=pca_df,
        legend='full',
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Outcome')
    plt.savefig(filename)
    plt.close()
    #plt.show()


if __name__ == "__main__":
    pca_sklearn(2)
    pca_sklearn(4)
    pca_sklearn(10)
    pca_manual(.95)
    pca_manual(.75)
    pca_manual(.50)
