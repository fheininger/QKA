import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import random
from collections import Counter
import numpy as original_numpy
# Helper for visualization and data creation
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# --- BUILDING BLOCKS FROM THE QSMOTE PAPER ---

def normalize_array(arr):
    """Normalizes a numpy array."""
    sum_of_squares = np.sum(np.square(arr))
    if np.isclose(sum_of_squares, 1.0):
        return arr
    scaling_factor = 1.0 / np.sqrt(sum_of_squares)
    return arr * scaling_factor

def calculate_angular_distance(data_point1, data_point2):
    """Calculates the angular distance based on the compact swap test formalism."""
    norm_dp1 = np.linalg.norm(data_point1)
    norm_dp2 = np.linalg.norm(data_point2)
    if norm_dp1 == 0 or norm_dp2 == 0:
        return np.pi
    z = norm_dp1**2 + norm_dp2**2
    inner_product_sq = ((norm_dp1**2 - norm_dp2**2)**2) / (2 * z)
    clipped_prob = np.clip(inner_product_sq, 0, 1)
    angular_distance = 2 * np.arccos(np.sqrt(clipped_prob))
    return angular_distance

def create_syn_data(n_qubits, angle_increment, angular_distance, data_point1, split_factor=10.0):
    """Generates a new synthetic data point by rotating a minority data point."""
    normalized_dp = normalize_array(data_point1)
    state_vector_size = 2**n_qubits
    if len(normalized_dp) < state_vector_size:
        padding = state_vector_size - len(normalized_dp)
        normalized_dp = np.pad(normalized_dp, (0, padding), 'constant')

    if angular_distance > np.pi / 2:
        angle = -angular_distance / split_factor
    elif angular_distance < 0:
        angle = (-angular_distance - random.uniform(0.5, 1.0)) / split_factor
    else:
        angle = random.uniform(0, angular_distance) / split_factor
    angle += angle_increment

    dev_rotate = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev_rotate)
    def rotation_circuit():
        qml.StatePrep(normalized_dp, wires=range(n_qubits))
        for i in range(n_qubits):
            qml.RX(angle, wires=i)
        return qml.state()

    new_statevector = rotation_circuit()
    new_data_point = np.real(new_statevector)
    return new_data_point[:len(data_point1)], angle

def create_synthetic_data_qsmotev2(minority_set, centroid_dp, n_qubits, num_samples_to_generate, split_factor=10.0):
    """Orchestrates the creation of synthetic data for the minority class."""
    minority_count = len(minority_set)
    if minority_count == 0 or num_samples_to_generate == 0:
        return pd.DataFrame(columns=minority_set.columns)

    synthetic_loop_itr = num_samples_to_generate // minority_count
    rem_synthetic_loop_itr = num_samples_to_generate % minority_count

    syn_data_list = []
    angular_distances = [calculate_angular_distance(row.to_numpy(), centroid_dp) for _, row in minority_set.iterrows()]
    minority_set_with_angles = minority_set.copy()
    minority_set_with_angles['angular_distance'] = angular_distances

    for i in range(synthetic_loop_itr):
        for _, dp_row in minority_set_with_angles.iterrows():
            dp = dp_row.drop('angular_distance').to_numpy()
            ang_dist = dp_row['angular_distance']
            angle_increment = i * 0.0174533
            syn_data, rot_angle = create_syn_data(n_qubits, angle_increment, ang_dist, dp, split_factor=split_factor)
            syn_record = {col: val for col, val in zip(minority_set.columns, syn_data)}
            syn_record['Rotation_angle'] = rot_angle
            syn_data_list.append(syn_record)

    if rem_synthetic_loop_itr > 0:
        remaining_minority_set = minority_set_with_angles.sample(n=rem_synthetic_loop_itr)
        for _, dp_row in remaining_minority_set.iterrows():
            dp = dp_row.drop('angular_distance').to_numpy()
            ang_dist = dp_row['angular_distance']
            angle_increment = synthetic_loop_itr * 0.0174533
            syn_data, rot_angle = create_syn_data(n_qubits, angle_increment, ang_dist, dp, split_factor=split_factor)
            syn_record = {col: val for col, val in zip(minority_set.columns, syn_data)}
            syn_record['Rotation_angle'] = rot_angle
            syn_data_list.append(syn_record)

    return pd.DataFrame(syn_data_list)

def generate_outliers(minority_df_orig, syn_data_frame, num_bins):
    """Identifies and bins angular outliers from the combined minority dataset."""
    if 'angular_distance' not in minority_df_orig.columns or (not syn_data_frame.empty and 'angular_distance' not in syn_data_frame.columns):
         raise ValueError("Input DataFrames must contain 'angular_distance' column.")
    minority_synthetic_df = pd.concat([minority_df_orig, syn_data_frame], ignore_index=True)
    q1 = minority_synthetic_df['angular_distance'].quantile(0.25)
    q3 = minority_synthetic_df['angular_distance'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_high = minority_synthetic_df[minority_synthetic_df['angular_distance'] > upper_bound]

    if outliers_high.empty:
        return pd.DataFrame(columns=['Bin_Start', 'Bin_End', 'Count'])
    counts, bin_edges = np.histogram(outliers_high['angular_distance'], bins=num_bins)
    return pd.DataFrame({'Bin_Start': bin_edges[:-1], 'Bin_End': bin_edges[1:], 'Count': counts})

def quantum_smote_boost(outlier_bins_df, smote_ds, n_qubits, num_bins, split_factor=10.0):
    """Boosts underrepresented outlier bins by generating more synthetic data."""
    if outlier_bins_df.empty:
        return pd.DataFrame()
    total_outlier_recs = outlier_bins_df['Count'].sum()
    if total_outlier_recs == 0:
        return pd.DataFrame()

    threshold = round(total_outlier_recs / num_bins) if num_bins > 0 else 0
    half_threshold = round(threshold / 2)
    boost_syn_data_list = []
    feature_cols = [col for col in smote_ds.columns if col not in ['angular_distance', 'Rotation_angle', 'Boosted']]

    for i, bin_row in outlier_bins_df.iterrows():
        if bin_row['Count'] < half_threshold and bin_row['Count'] > 0:
            minority_temp = smote_ds[(smote_ds['angular_distance'] >= bin_row['Bin_Start']) & (smote_ds['angular_distance'] < bin_row['Bin_End'])]
            if minority_temp.empty:
                continue
            synthetic_loop_itr = int(np.floor(threshold / bin_row['Count']))
            for j, dp_row in minority_temp.iterrows():
                minority_dp_temp = dp_row[feature_cols].to_numpy()
                for k in range(synthetic_loop_itr):
                    angle_increment = (k * 0.0174533) * 1.5 + (j % minority_temp.shape[0]) * 0.001
                    angular_distance = dp_row['angular_distance']
                    boost_split_factor = max(1.0, split_factor / 1.5)
                    boost_syn_data, rot_angle = create_syn_data(n_qubits, angle_increment, angular_distance, minority_dp_temp, split_factor=boost_split_factor)
                    syn_record = {col: val for col, val in zip(feature_cols, boost_syn_data)}
                    syn_record['Rotation_angle'] = rot_angle
                    syn_record['Boosted'] = 'Yes'
                    boost_syn_data_list.append(syn_record)
    return pd.DataFrame(boost_syn_data_list)


# --- The Main Wrapper Function You Will Call ---
# MODIFIED FUNCTION SIGNATURE: Added n_qubits parameter
def quantum_smote(features, labels, n_qubits, desired_ratio=1.0, split_factor=10.0, boost_outliers=True, num_bins_for_outliers=10):
    """
    Applies the Quantum-SMOTEV2 algorithm to an imbalanced dataset.

    Args:
        features (np.ndarray): The training feature data (xs_train).
        labels (np.ndarray): The training labels (y_train).
        n_qubits (int): The number of qubits to use, determined by the number of features.
        desired_ratio (float): The desired ratio of minority to majority samples. Defaults to 1.0.
        split_factor (float): Divisor for the rotation angle. Smaller value = more spread. Defaults to 10.0.
        boost_outliers (bool): If True, applies Angular Outlier Boosting.
        num_bins_for_outliers (int): The number of bins for identifying under-represented outlier regions.

    Returns:
        tuple (np.ndarray, np.ndarray): A tuple containing the resampled features and labels.
    """
    # --- New validation step ---
    num_features = features.shape[1]
    if num_features > 2**n_qubits:
        raise ValueError(
            f"The number of features ({num_features}) cannot be represented by {n_qubits} qubits. "
            f"Required qubits: {int(np.ceil(np.log2(num_features)))}. Or reduce features to <= {2**n_qubits}."
        )

    class_counts = Counter(labels)
    if len(class_counts) != 2:
        raise ValueError(f"Quantum-SMOTE is for binary classification, but found {len(class_counts)} classes.")

    minority_class_label = min(class_counts, key=class_counts.get)
    majority_class_label = max(class_counts, key=class_counts.get)
    minority_count = class_counts[minority_class_label]
    majority_count = class_counts[majority_class_label]

    if minority_count >= (majority_count * desired_ratio):
        print("Dataset is already balanced. No oversampling needed.")
        return features, labels

    num_samples_to_generate = int(majority_count * desired_ratio) - minority_count
    if num_samples_to_generate <= 0:
        print("Desired ratio is already met. No oversampling needed.")
        return features, labels

    features_df = pd.DataFrame(features)
    labels_series = pd.Series(labels)
    minority_df = features_df[labels_series == minority_class_label]
    centroid = features_df.mean().to_numpy()

    print(f"Generating {num_samples_to_generate} new minority samples with {n_qubits} qubits and split_factor={split_factor}...")
    synthetic_df = create_synthetic_data_qsmotev2(
        minority_set=minority_df,
        centroid_dp=centroid,
        n_qubits=n_qubits,
        num_samples_to_generate=num_samples_to_generate,
        split_factor=split_factor
    )

    boosted_df = pd.DataFrame()
    if boost_outliers:
        print("Performing Angular Outlier Boosting...")
        original_minority_with_angles = minority_df.copy()
        original_minority_with_angles['angular_distance'] = [calculate_angular_distance(row.to_numpy(), centroid) for _, row in minority_df.iterrows()]
        if not synthetic_df.empty:
            synthetic_with_angles = synthetic_df.copy()
            feature_cols_syn = [col for col in synthetic_df.columns if col != 'Rotation_angle']
            synthetic_with_angles['angular_distance'] = [calculate_angular_distance(row.to_numpy(), centroid) for _, row in synthetic_df[feature_cols_syn].iterrows()]
        else:
            synthetic_with_angles = pd.DataFrame(columns=original_minority_with_angles.columns)
        outlier_bins = generate_outliers(original_minority_with_angles, synthetic_with_angles, num_bins_for_outliers)
        all_minority_data = pd.concat([original_minority_with_angles, synthetic_with_angles], ignore_index=True)
        boosted_df = quantum_smote_boost(outlier_bins, all_minority_data, n_qubits, num_bins_for_outliers, split_factor=split_factor)
        print(f"Generated {len(boosted_df)} boosted outlier samples.")

    if not synthetic_df.empty:
        feature_cols = minority_df.columns
        xs_synthetic = synthetic_df[feature_cols].to_numpy()
        y_synthetic = np.full(len(xs_synthetic), minority_class_label)
        features = np.vstack((features, xs_synthetic))
        labels = np.hstack((labels, y_synthetic))

    if not boosted_df.empty:
        feature_cols = minority_df.columns
        xs_boosted = boosted_df[feature_cols].to_numpy()
        y_boosted = np.full(len(xs_boosted), minority_class_label)
        features = np.vstack((features, xs_boosted))
        labels = np.hstack((labels, y_boosted))
     # --- FINAL FIX: Convert labels back to a standard NumPy int array ---
    # This prevents the "tensor" type from leaking into your classical model.
    final_labels = original_numpy.array(labels, dtype=int)
    
    return features, final_labels

# --- EXAMPLE USAGE BLOCK ---

def visualize_data(xs, ys, title):
    """A simple helper function to visualize 2D data for demonstration."""
    # This visualization is only meaningful if the number of features is 2.
    # If more, we use PCA just for plotting purposes.
    if xs.shape[1] > 2:
        plot_pca = PCA(n_components=2)
        xs_2d = plot_pca.fit_transform(xs)
    else:
        xs_2d = xs

    plt.figure(figsize=(8, 6))
    plt.scatter(xs_2d[ys==0][:, 0], xs_2d[ys==0][:, 1], label="Class 0 (Majority)")
    plt.scatter(xs_2d[ys==1][:, 0], xs_2d[ys==1][:, 1], label="Class 1 (Minority)", alpha=0.7, marker="x")
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 1. Create a sample imbalanced dataset with many features
    xs_train, y_train = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=8,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42
    )
    print(f"Original data shape: {xs_train.shape}")
    print("Original class distribution:", Counter(y_train))

    # 2. Scale data and apply PCA to reduce dimensionality
    scaler = StandardScaler()
    xs_scaled = scaler.fit_transform(xs_train)

    N_COMPONENTS = 4 # Decide how many principal components to keep
    pca = PCA(n_components=N_COMPONENTS)
    xs_pca = pca.fit_transform(xs_scaled)
    print(f"\nData shape after PCA: {xs_pca.shape}")

    # 3. Calculate the number of qubits needed for the new dimensionality
    # This is the value you must pass to the function.
    N_QUBITS = int(np.ceil(np.log2(N_COMPONENTS)))
    print(f"Number of qubits required for {N_COMPONENTS} features: {N_QUBITS}")

    # Visualize the PCA-reduced data
    visualize_data(xs_pca, y_train, f"After PCA ({N_COMPONENTS} Components)")

    # 4. Apply Quantum-SMOTE
    xs_resampled, y_resampled = quantum_smote(
        features=xs_pca,
        labels=y_train,
        n_qubits=N_QUBITS, # Pass the calculated number of qubits
        desired_ratio=1.0,
        split_factor=0.001 # Use a smaller factor for more spread
    )

    print(f"\nClass distribution after Q-SMOTE: {Counter(y_resampled)}")
    visualize_data(xs_resampled, y_resampled, f"After Q-SMOTE ({N_COMPONENTS} Components)")
