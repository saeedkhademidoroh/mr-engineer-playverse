# Third-party imports
import numpy as np
import tensorflow as tf
from keras.api.utils import to_categorical

# Function to load dataset
def load_dataset():
    """
    Loads MNIST dataset and returns train and test splits.

    Returns:
        tuple: (train_data, train_labels), (test_data, test_labels) as NumPy arrays.
    """


    return tf.keras.datasets.mnist.load_data()


# Function to analyze dataset (statistical analysis)
def analyze_dataset(train_data, train_labels, test_data, test_labels):
    """
    Performs statistical analysis of dataset, including:
    - Shape and data types
    - Missing values
    - Summary statistics for pixel intensity distribution

    Parameters:
        train_data (numpy.ndarray): Training feature set.
        test_data (numpy.ndarray): Testing feature set.
        train_labels (numpy.ndarray): Training labels.
        test_labels (numpy.ndarray): Testing labels.

    Prints:
        - Dataset shape and data types.
        - Count of missing values.
        - Statistical summary of pixel intensities (min, max, mean, std).
    """


    # Print header for function
    print("\n🎯 Dataset Analysis 🎯")

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

    # Compute statistics across all images
    min_values = []
    max_values = []
    mean_values = []
    std_values = []

    for i in range(train_data.shape[0]): # Iterate over all images
        image = train_data[i].flatten()
        min_values.append(image.min())
        max_values.append(image.max())
        mean_values.append(image.mean())
        std_values.append(image.std())

    # Compute overall mean statistics
    mean_min = np.mean(min_values)
    mean_max = np.mean(max_values)
    mean_mean = np.mean(mean_values)
    mean_std = np.mean(std_values)

    # Summary Statistics
    print("\n🔹 Statistical Summary (Pixel Values):\n")
    print(f"Extreme-Range: {train_data.min()} to {train_data.max()}")
    print(f"Mean-Min: {mean_min}")
    print(f"Mean-Max: {mean_max}")
    print(f"Mean-Mean: {mean_mean}")
    print(f"Mean-Std: {mean_std}")


# Function to preprocess dataset (normalization, reshaping, etc.)
def preprocess_dataset(train_data, train_labels, test_data, test_labels):
    """
    Preprocesses MNIST dataset by:
    - Flattening image data.
    - Normalizing pixel values to range [0,1].
    - Converting labels to one-hot encoding.
    - Splitting validation set from training data.

    Parameters:
        train_data (numpy.ndarray): Training images.
        test_data (numpy.ndarray): Testing images.
        train_labels (numpy.ndarray): Training labels.
        test_labels (numpy.ndarray): Testing labels.

    Returns:
        tuple: Preprocessed (train_data, train_labels, test_data, test_labels, val_data, val_labels).
    """


    # Print header for function
    print("\n🎯 Preprocess Dataset 🎯")

    # Print data types before preprocessing
    print("\n🔹 Data Types Before Preprocessing:\n")
    print("Train Data Type:", train_data.dtype)
    print("Test Data Type:", test_data.dtype)
    print("Train Labels Type:", train_labels.dtype)
    print("Test Labels Type:", test_labels.dtype)

    # Print shapes before preprocessing
    print("\n🔹 Data Shapes Before Preprocessing:\n")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    print("Train Labels Shape:", train_labels.shape)
    print("Test Labels Shape:", test_labels.shape)

    # Reshape data to ensure compatibility
    train_data = np.reshape(train_data, (-1, 28 * 28))
    test_data = np.reshape(test_data, (-1, 28 * 28))

    # Normalize data with dividing by 255 and convert to float32
    train_data = (train_data / 255.0).astype(np.float32)
    test_data = (test_data / 255.0).astype(np.float32)

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Split data into training and validation sets
    train_data, val_data = train_data[:30000], train_data[30000:40000]
    train_labels, val_labels = train_labels[:30000], train_labels[30000:40000]

    # Print data types after preprocessing
    print("\n🔹 Data Types After Preprocessing:\n")
    print("Train Data Type:", train_data.dtype)
    print("Test Data Type:", test_data.dtype)
    print("Train Labels Type:", train_labels.dtype)
    print("Test Labels Type:", test_labels.dtype)

    # Print shapes after preprocessing
    print("\n🔹 Data Shapes After Preprocessing:\n")
    print("Train Data Shape:", train_data.shape)
    print("Test Data Shape:", test_data.shape)
    print("Train Labels Shape:", train_labels.shape)
    print("Test Labels Shape:", test_labels.shape)

    return train_data, train_labels, test_data, test_labels, val_data, val_labels


# Print confirmation message
print("\n✅ data.py successfully executed")