# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Function to load dataset
def load_dataset():
    """
    Loads and returns Boston Housing dataset as training and test splits.

    Returns:
        tuple: A tuple containing training data, training labels, test data, and test labels
    """


    return tf.keras.datasets.boston_housing.load_data()


# Function to analyze dataset (statistical analysis)
def analyze_dataset(train_data, train_labels, test_data, test_labels):
    """
    Performs statistical analysis on given dataset, including:
    - Shape and data types
    - Missing values
    - Summary statistics

    Parameters:
        train_data (numpy.ndarray): Training feature set
        train_labels (numpy.ndarray): Training labels
        test_data (numpy.ndarray): Testing feature set
        test_labels (numpy.ndarray): Testing labels
    """


    # Print header for function
    print("\n🎯 Dataset Analysis 🎯")

    # Convert to DataFrame for better analysis
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    train_labels_df = pd.DataFrame(train_labels, columns=[''])
    test_labels_df = pd.DataFrame(test_labels, columns=[''])

    # Dataset Shape and Data Types
    print("\n🔹 Dataset Shape & Data Types:\n")
    print(f"Train data shape: {train_data.shape}, Type: {train_data.dtype}")
    print(f"Test data shape: {test_data.shape}, Type: {test_data.dtype}")
    print(f"Train labels shape: {train_labels.shape}, Type: {train_labels.dtype}")
    print(f"Test labels shape: {test_labels.shape}, Type: {test_labels.dtype}")

    # Checking for Missing Values
    print("\n🔹 Missing Values:\n")
    print(f"Train data missing values: {np.isnan(train_data).sum()}")
    print(f"Test data missing values: {np.isnan(test_data).sum()}")
    print(f"Train labels missing values: {np.isnan(train_labels).sum()}")
    print(f"Test labels missing values: {np.isnan(test_labels).sum()}")

    # Summary Statistics
    print("\n🔹 Statistical Summary:\n")
    print("Train Data", train_df.describe())
    print("\nTest Data", test_df.describe())
    print("\nTrain Labels", train_labels_df.describe())
    print("\nTest Labels", test_labels_df.describe())


# Function to preprocess dataset (normalization, reshaping, etc.)
def preprocess_dataset(train_data, train_labels, test_data, test_labels):
    """
    Preprocesses dataset for models by:
    - Reshaping labels
    - Normalizing data using MinMaxScaler
    - Printing pre and post normalization data ranges
    - Converting data types to float32 for optimization

    Parameters:
        train_data (numpy.ndarray): Training feature set
        train_labels (numpy.ndarray): Training labels
        test_data (numpy.ndarray): Testing feature set
        test_labels (numpy.ndarray): Testing labels

    Returns:
        tuple: A tuple containing scaled and reshaped training and test data and labels
    """


    # Print header for function
    print("\n🎯 Preprocessing Steps 🎯")

    print("\n🔹 Before Reshaping:\n")
    print("Train Labels Shape:", train_labels.shape)
    print("Test Labels Shape:", test_labels.shape)

    # Reshape labels to ensure compatibility
    train_labels = np.reshape(train_labels, (-1, 1))
    test_labels = np.reshape(test_labels, (-1, 1))

    print("\n🔹 After Reshaping:\n")
    print("Train Labels Shape:", train_labels.shape)
    print("Test Labels Shape:", test_labels.shape)

    # Check label ranges
    train_labels_min, train_labels_max = train_labels.min(axis=0), train_labels.max(axis=0)
    test_labels_min, test_labels_max = test_labels.min(axis=0), test_labels.max(axis=0)

    print("\n🔹 Label Ranges:\n")
    print("Train Labels Min:", train_labels_min, "\nTrain Labels Max:", train_labels_max)
    print("Test Labels Min:", test_labels_min, "\nTest Labels Max:", test_labels_max)

    # Check pre-normalization data ranges
    train_data_min, train_data_max = train_data.min(axis=0), train_data.max(axis=0)
    test_data_min, test_data_max = test_data.min(axis=0), test_data.max(axis=0)

    print("\n🔹 Pre-Normalization Data Ranges:\n")
    print("Train Data Min:", train_data_min, "\nTrain Data Max:", train_data_max)
    print("Test Data Min:", test_data_min, "\nTest Data Max:", test_data_max)

    # Fit scaler on training data only
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(train_data)

    # Transform both training and test data using scaler
    train_data = min_max_scaler.transform(train_data)
    test_data = min_max_scaler.transform(test_data)

    # Check post-normalization min/max values
    train_min_post, train_max_post = train_data.min(axis=0), train_data.max(axis=0)
    test_min_post, test_max_post = test_data.min(axis=0), test_data.max(axis=0)

    print("\n🔹 Post-Normalization Data Ranges:\n")
    print("Post-Normalization Train Data Min:", train_min_post, "\nPost-Normalization Train Data Max:", train_max_post)
    print("Post-Normalization Test Data Min:", test_min_post, "\nPost-Normalization Test Data Max:", test_max_post)

    print("\n🔹 Data Types Before Conversion:\n")
    print("Train Data Type:", train_data.dtype)
    print("Test Data Type:", test_data.dtype)
    print("Train Labels Type:", train_labels.dtype)
    print("Test Labels Type:", test_labels.dtype)

    # Convert dataset values to float32 for optimization
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    print("\n🔹 Data Types After Conversion:\n")
    print("Train Data Type:", train_data.dtype)
    print("Test Data Type:", test_data.dtype)
    print("Train Labels Type:", train_labels.dtype)
    print("Test Labels Type:", test_labels.dtype)

    return train_data, train_labels, test_data, test_labels


# Print confirmation message
print("\n✅ data.py successfully executed")