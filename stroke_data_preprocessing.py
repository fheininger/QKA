import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

le = LabelEncoder()

def preprocess_stroke_dataset(data, normalize=False, scale=False, apply_pca=False, N=None, test_size=0.3):
    """
    Preprocessing of the stroke dataset including:
    - Handling missing values (Mean/Median Imputation)
    - Encoding categorical features
    - Normalization, scaling, and PCA (optional)
    - Train-test split

    Parameters:
        data (pd.DataFrame): The input dataset.
        normalize (bool): Apply normalization.
        scale (bool): Apply Min-Max scaling.
        apply_pca (bool): Apply PCA for dimensionality reduction.
        N (int): Number of PCA components (only used if apply_pca=True).
        test_size (float): Proportion of data to use as the test set.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        Y_train (pd.Series): Training target labels.
        Y_test (pd.Series): Test target labels.
    """
    print("Missing values before processing:\n", data.isna().sum())

    # Fill numerical missing values
    imputer_mean = SimpleImputer(strategy="mean")  # Mean for normally distributed data
    imputer_median = SimpleImputer(strategy="median")  # Median for skewed/outlier data

    data["bmi"] = imputer_mean.fit_transform(data[["bmi"]])  # Mean imputation
    data["avg_glucose_level"] = imputer_median.fit_transform(data[["avg_glucose_level"]])  # Median imputation

    print("Duplicate rows:", data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    data.drop(columns=['id'], inplace=True)
    data = data[data["gender"] != "Other"]  # Remove "Other" gender category
    
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    X = data.drop(columns=['stroke'])
    Y = data['stroke']
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=42)
    
    if normalize:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    if scale:
        min_max_scaler = MinMaxScaler()
        X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(min_max_scaler.transform(X_test), columns=X.columns)

    if apply_pca:
        if N is None or N > X.shape[1]:  
            raise ValueError(f"PCA components (N={N}) must be specified and ≤ {X.shape[1]}.")
        pca = PCA(n_components=N)
        X_train = pd.DataFrame(pca.fit_transform(X_train))
        X_test = pd.DataFrame(pca.transform(X_test))
        exp_var_pca = pca.explained_variance_ratio_
        print(np.cumsum(exp_var_pca))
    print(f"Final dataset shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")
    return X_train, X_test, Y_train, Y_test

