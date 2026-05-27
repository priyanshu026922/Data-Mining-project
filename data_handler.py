import pandas as pd
import numpy as np

def read_one_file(file_path):
    """
    Reads one CSV file
    Removes first 2 columns
    Returns features and labels separately
    """
    
    data = pd.read_csv(file_path)
    
    print("File shape:", data.shape)
    print("Columns:", data.columns.tolist())
    
    
    features = data.iloc[:, 2:-1].values    # All rows, columns 2 to second-last
    labels = data.iloc[:, -1].values        # All rows, last column only
    
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Sample labels:", labels[:20])     #First 20 labels
    
    return features, labels

def create_sliding_windows(features, labels, window_size=50, step_size=25):
    
    windows = []
    window_labels = []
    num_rows = len(labels)
    start = 0
    
    while start + window_size <= num_rows:
        end = start + window_size

        window_features = features[start:end, :]

        window_labs = labels[start:end]

        windows.append(window_features)

        count_ones = np.sum(window_labs == 1)

        if count_ones > 0.4 * window_size:
            window_labels.append(1)
        else:
            window_labels.append(0)
            
        start = start + step_size
    
    
    #Handle leftover rows at the end
    if start < num_rows:
        final_start = num_rows - window_size  

        window_features = features[final_start:, :]

        window_labs = labels[final_start:]

        windows.append(window_features)

        count_ones = np.sum(window_labs == 1)

        if count_ones > 0.4 * window_size:
            window_labels.append(1)
        else:
            window_labels.append(0)

        print(f"Added final window from row {final_start} to {num_rows}")
    
    print(f"Created {len(windows)} windows from {num_rows} rows")

    return windows, window_labels


def flatten_windows(windows):
    """
    Converts list of 2D windows to 2D array
    
    Input:
        windows: list of arrays, each (50, 9)
    
    Output:
        flattened: numpy array of shape (num_windows, 450)
    """
    
    flattened = []
    
    for window in windows:
        # Flatten 50×9 to 450
        flat = window.flatten()
        flattened.append(flat)
    
    # Convert list to numpy array
    flattened = np.array(flattened)
    
    print(f"Flattened shape: {flattened.shape}")
    
    return flattened

import os


def process_one_folder(folder_path):
    """
    Processes all CSV files in one folder
    Returns combined windows and labels from all files
    """
    
    all_windows = []
    all_labels = []
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Keep only CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files in {folder_path}")
    
    for file_name in csv_files:
        
        file_path = os.path.join(folder_path, file_name)
        
        # Read file
        features, labels = read_one_file(file_path)
        
        # Check if file has enough rows for at least one window
        if len(labels) < 50:
            print(f"  Skipping {file_name}: only {len(labels)} rows (need 50)")
            continue
        
        # Create windows
        windows, window_labels = create_sliding_windows(features, labels)
        
        # Add to collection
        all_windows.extend(windows)
        all_labels.extend(window_labels)
    
    print(f"Total windows from this folder: {len(all_windows)}")
    
    return all_windows, all_labels



def process_all_training_data(main_folder_path):
    """
    Processes all subfolders inside the main training folder
    Returns X (features) and y (labels) for entire training set
    """
    
    all_windows = []
    all_labels = []
    
    # List all subfolders
    subfolders = os.listdir(main_folder_path)
    
    print(f"Found {len(subfolders)} subfolders")
    print("-" * 50)
    
    for subfolder_name in subfolders:
        
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        
    
        if not os.path.isdir(subfolder_path):#if not a folder
            continue
        
        print(f"\nProcessing: {subfolder_name}")
        
        # Process this subfolder
        windows, labels = process_one_folder(subfolder_path)
        
        # Add to collection
        all_windows.extend(windows)
        all_labels.extend(labels)
    
    print("\n" + "=" * 50)
    print(f"TOTAL windows from all folders: {len(all_windows)}")
    
    # Flatten all windows
    X = flatten_windows(all_windows)
    y = np.array(all_labels)
    
    print(f"\nFINAL Training Data:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class 0: {sum(y == 0)} ({100*sum(y==0)/len(y):.1f}%)") ##in percentage
    print(f"Class 1: {sum(y == 1)} ({100*sum(y==1)/len(y):.1f}%)")
    
    return X, y


def process_all_testing_data(main_folder_path):
    """
    Same as training, but for test data
    """
    
    all_windows = []
    all_labels = []
    
    subfolders = os.listdir(main_folder_path)
    
    print(f"Found {len(subfolders)} test subfolders")
    print("-" * 50)
    
    for subfolder_name in subfolders:
        
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        print(f"\nProcessing: {subfolder_name}")
        
        windows, labels = process_one_folder(subfolder_path)
        
        all_windows.extend(windows)
        all_labels.extend(labels)
    
    print("\n" + "=" * 50)
    print(f"TOTAL test windows: {len(all_windows)}")
    
    X = flatten_windows(all_windows)
    y = np.array(all_labels)
    
    print(f"\nFINAL Test Data:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class 0: {sum(y == 0)}")
    print(f"Class 1: {sum(y == 1)}")
    
    return X, y



if __name__ == "__main__":
    
    training_folder = "Data/Sample_training"
    
    X_train, y_train = process_all_training_data(training_folder)
    
    print("\nTraining AND Testing  data is ready!")

