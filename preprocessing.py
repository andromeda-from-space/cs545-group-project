import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(test_data, test_label, train_data, train_label):
    # Load the dataset
    data = pd.read_csv('dataset.csv')
    print(f"Dataset shape: {data.shape}")
    
    # Handle missing values
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Split features and target
    X = data.drop(columns=['Target'])
    y = data['Target']
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    print(f"Target classes: {target_encoder.classes_}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Convert targets to one-hot encoding
    train_label = np.zeros((y_train_balanced.shape[0], len(np.unique(y))))
    for i in range(0, y_train_balanced.shape[0]):
        train_label[i, y_train_balanced[i]] = 1
    
    test_label = np.zeros((y_test.shape[0], len(np.unique(y))))
    for i in range(0, y_test.shape[0]):
        test_label[i, y_test[i]] = 1
    
    # Assign the preprocessed data to function parameters
    train_data = X_train_balanced
    test_data = X_test
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training labels shape: {train_label.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_label.shape}")
    
    return train_data, train_label, test_data, test_label

def visualize_data(train_data, train_label, test_data, test_label):
    """
    Create visualizations to understand the data
    """
    # Visualize class distribution
    class_counts = np.sum(train_label, axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_counts)), class_counts)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.title('Class Distribution in Training Data')
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Visualize feature distributions
    plt.figure(figsize=(12, 8))
    for i in range(min(5, train_data.shape[1])):  # Plot first 5 features
        plt.subplot(2, 3, i+1)
        plt.hist(train_data[:, i], bins=20)
        plt.title(f'Feature {i}')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    return

def feature_importance(train_data, train_label):
    """
    Calculate feature importance using Random Forest
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Convert one-hot encoded labels back to single column
    y_train = np.argmax(train_label, axis=1)
    
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_data, y_train)
    
    # Get feature importance
    importances = rf.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(min(10, train_data.shape[1])):  # Print top 10 features
        print(f"{f+1}. Feature {indices[f]} ({importances[indices[f]]})")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(min(20, train_data.shape[1])), importances[indices[:20]], align='center')
    plt.xticks(range(min(20, train_data.shape[1])), indices[:20])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()
    
    return

# Add this function to perform additional preprocessing steps
def additional_preprocessing(train_data, train_label, test_data, test_label):
    """
    Perform additional preprocessing steps
    """
    # Create feature interactions (example for educational data)
    # For example, create a new feature that combines academic performance across semesters
    if train_data.shape[1] >= 10:  # Ensure we have enough features
        # Assuming feature 5 and 6 are academic grades from different semesters
        new_feature_train = train_data[:, 5] * train_data[:, 6]
        new_feature_test = test_data[:, 5] * test_data[:, 6]
        
        # Add the new feature to the data
        train_data = np.column_stack((train_data, new_feature_train.reshape(-1, 1)))
        test_data = np.column_stack((test_data, new_feature_test.reshape(-1, 1)))
        
        print(f"Added interaction feature. New data shape: {train_data.shape}")
    '''
    # Perform Principal Component Analysis for dimensionality reduction (optional)
    from sklearn.decomposition import PCA
    
    if train_data.shape[1] > 20:  # If we have many features
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        train_data_pca = pca.fit_transform(train_data)
        test_data_pca = pca.transform(test_data)
        
        print(f"Applied PCA. Reduced dimensions from {train_data.shape[1]} to {train_data_pca.shape[1]}")
        return train_data_pca, train_label, test_data_pca, test_label
    '''
    return train_data, train_label, test_data, test_label


# Add this function for MNIST data preprocessing
def load_mnist_data(test_data, test_label, train_data, train_label):
    """
    Load and preprocess MNIST data from Kaggle paths
    """
    # Pre-process the training data
    train_data = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
    perm = np.array(range(0, train_data.shape[0]))
    perm = np.random.permutation(perm)
    train_data = train_data[perm, :]
    train_label = np.zeros([train_data.shape[0], 10])
    for i in range(0, train_data.shape[0]):
        train_label[i, int(train_data[i, 0])] = 1
    train_data = train_data / 255
    train_data[:, 0] = 1
    
    # Pre-process the test data
    test_data = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1)
    perm = np.array(range(0, test_data.shape[0]))
    perm = np.random.permutation(perm)
    test_data = test_data[perm, :]
    test_label = np.zeros([test_data.shape[0], 10])
    for i in range(0, test_data.shape[0]):
        test_label[i, int(test_data[i, 0])] = 1
    test_data = test_data / 255
    test_data[:, 0] = 1
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training labels shape: {train_label.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_label.shape}")
    
    return train_data, train_label, test_data, test_label

def main():
    # Initialize empty arrays
    train_data = None
    train_label = None
    test_data = None
    test_label = None
    
    # Choose which dataset to use (uncomment one option)
    
    # Option 1: Load and preprocess student dropout data
    train_data, train_label, test_data, test_label = load_data(test_data, test_label, train_data, train_label)
    
    # Option 2: Load and preprocess MNIST data
    #train_data, train_label, test_data, test_label = load_mnist_data(test_data, test_label, train_data, train_label)
    
    # Visualize the data
    visualize_data(train_data, train_label, test_data, test_label)
    
    # Calculate feature importance
    feature_importance(train_data, train_label)
    
    # Apply additional preprocessing steps
    train_data, train_label, test_data, test_label = additional_preprocessing(train_data, train_label, test_data, test_label)
    
    print("Preprocessing completed successfully!")
    
    return train_data, train_label, test_data, test_label

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = main()