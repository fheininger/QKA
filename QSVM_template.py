import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
# --- Repository paths ------------------------------------------------------
# Datasets ship inside the repo (./data). Results are written to $QSVM_RESULTS,
# defaulting to ~/results. Both are overridable so nothing is machine-specific.
from pathlib import Path as _Path

import matplotlib.colors as colors
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from matplotlib import font_manager
from matplotlib.lines import Line2D
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Imported up here (not next to the Ising block) because the circuit topology it
# defines decides how many qubits the dimensionality-reduction branches need.
import kernel_align
from plane_imports import *
from quantum_smote import quantum_smote
from roc_results import record_roc_comparison
from stroke_data_preprocessing import *

DATA_DIR = _Path(os.environ.get("QSVM_DATA", _Path(__file__).resolve().parent / "data"))
RESULTS = _Path(os.environ.get("QSVM_RESULTS", _Path.home() / "results"))

N_QUBITS = 7  # q in the manuscript; SVD components = qubit-register size
oversampling = "SMOTE"  # Options: "QSMOTE", "SMOTE", "NONE"
folder_roc = str(RESULTS / "roc_curves")
folder_kernel = str(RESULTS / "kernel_matrices")
os.makedirs(folder_roc, exist_ok=True)
os.makedirs(folder_kernel, exist_ok=True)


def bootstrap_roc(y_true, y_score, n_boot=1000, seed=42):
    """Bootstrap the test-set ROC curve.

    Returns (mean_fpr, mean_tpr, std_tpr, boot_aucs) on a fixed 100-point FPR grid.
    Resamples with replacement; draws that end up single-class are skipped.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[idx], y_score[idx])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    return mean_fpr, np.mean(tprs, axis=0), np.std(tprs, axis=0), np.array(aucs)


# Arial is not installed on every machine; Liberation Sans has identical metrics.
# Resolved once here so matplotlib doesn't warn "findfont: Arial not found" on every label.
_installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
ROC_LABEL_FONT = dict(
    fontsize=14,
    fontfamily=next(
        (name for name in ("Arial", "Liberation Sans") if name in _installed_fonts),
        "sans-serif",
    ),
)


def table_legend(ax, handles, names, values, title=None):
    """Lower-right legend laid out as two columns, so the values line up under each other.

    Two frameless legends with identical row spacing: the values one sits in the corner,
    the names one is anchored to its left edge. Call after tight_layout -- the anchor is
    measured in axes coordinates, so a later layout change would shift the columns apart.
    """
    blank = [Line2D([], [], linestyle="none") for _ in values]
    value_legend = ax.legend(
        blank,
        values,
        loc="lower right",
        frameon=False,
        handlelength=0,
        handletextpad=0,
        title=title,
        alignment="left",
    )
    ax.add_artist(value_legend)
    ax.figure.canvas.draw()
    x0, y0 = ax.transAxes.inverted().transform(value_legend.get_window_extent().p0)
    ax.legend(
        handles,
        names,
        loc="lower right",
        bbox_to_anchor=(x0, y0),
        borderaxespad=0,
        frameon=False,
    )


def visualize_data(xs, ys, title):
    """
    A helper function to visualize high-dimensional data by reducing it to 2D using PCA.
    """
    # If data is already 2D, just use it. Otherwise, apply PCA for plotting.
    if xs.shape[1] > 2:
        plot_pca = PCA(n_components=2)
        xs_2d = plot_pca.fit_transform(xs)
        plot_title = f"{title} (Visualized with PCA)"
    else:
        xs_2d = xs
        plot_title = title

    plt.figure(figsize=(8, 6))
    plt.scatter(xs_2d[ys == 0][:, 0], xs_2d[ys == 0][:, 1], label="Class 0", alpha=0.6)
    plt.scatter(
        xs_2d[ys == 1][:, 0],
        xs_2d[ys == 1][:, 1],
        label="Class 1",
        alpha=0.8,
        marker="x",
    )
    plt.title(plot_title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


FIXING = True 
percent = 0.5
TEST_SIZE = 0.3
jobs = -1
if FIXING is False:
    percent = None
save_fig = True
align = True
# Kernel/alignment switches. KERNEL_MODE="exact" builds every Gram from feature
# states (|<phi(a)|phi(b)>|^2) instead of one circuit per entry: ~560x faster and
# exact rather than shot-sampled. "shots" restores the legacy per-entry path.
# ALIGN_ONLY skips the baseline pipeline so the search can be run on its own.
# Do NOT use GRID_SEARCH=False for that -- line 2589 reads `grid` outside its guard.
KERNEL_MODE = "exact"  # "exact" | "shots"
# Qubit topology the features are loaded onto. The Ising couplings sit on
# nearest-neighbour pairs, so an open CHAIN of n qubits carries only n-1 features
# and silently drops the last one -- with N_QUBITS = N_COMPONENTS = 7 the 7th PCA
# component never entered the circuit. "ring" closes the chain with the (n-1, 0)
# pair and loads all n. This changes the kernel: nothing computed under "chain"
# is comparable to a "ring" number, so the outputs are tagged separately.
TOPOLOGY = "ring"  # "ring" | "chain" (chain reproduces pre-fix results)
kernel_align.TOPOLOGY = TOPOLOGY
ALIGN_ONLY = False  # True: run the alignment search only
ALIGN_PRESET = None  # [t, T, L] from an earlier search; skips searching
ALIGN_OUT = str(RESULTS / "alignment")
roc_multiclass = "raise"
### Test- unfortunately, I am little bit stupid
"""
At this point, it might be useful to make it possible to use different datasets. Interface is not intended, but some datasets might require different preprocessing steps.
"""

dataset = "pdac"

if dataset == "stroke":

    # Stroke Data preprocessing
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import (LabelEncoder, MinMaxScaler,
                                       StandardScaler)

    # Load your dataset
    data = pd.read_csv(DATA_DIR / "healthcare-dataset-stroke-data.csv")
    # === Step 1: Report missing values ===
    print("Missing values before processing:\n", data.isna().sum())

    # === Step 2: Impute missing values ===
    imputer_mean = SimpleImputer(strategy="mean")
    imputer_median = SimpleImputer(strategy="median")

    data["bmi"] = imputer_mean.fit_transform(data[["bmi"]])
    data["avg_glucose_level"] = imputer_median.fit_transform(
        data[["avg_glucose_level"]]
    )

    # === Step 3: Remove duplicates and drop ID column ===
    print("Duplicate rows:", data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    data.drop(columns=["id"], inplace=True)

    # === Step 4: Remove invalid gender category ===
    data = data[data["gender"] != "Other"]

    # === Step 5: Encode categorical columns ===
    le = LabelEncoder()
    categorical_cols = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # === Step 6: Split features and target ===
    X = data.drop(columns=["stroke"])
    Y = data["stroke"]
    # show full sample size
    print(f"Full dataset shape: {X.shape}, Target shape: {Y.shape}")
    # === For Fixing: take smaller part of dataset ===
    if FIXING is True:
        X, _, Y, _ = train_test_split(
            X, Y, train_size=percent, stratify=Y, random_state=42
        )

    print(f"Full dataset shape after scimming: {X.shape}, Target shape: {Y.shape}")
    # === Step 7: Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y, random_state=42
    )

    # === Step 8 (Optional): Standard normalization ===
    # scaler = StandardScaler()
    # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    # X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # === Step 9 (Optional): Min-Max scaling ===
    min_max_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(min_max_scaler.transform(X_test), columns=X.columns)
    N_COMPONENTS = N_QUBITS
    # pca = PCA(n_components=N_QUBITS)
    pca = TruncatedSVD(n_components=N_QUBITS)
    xs_train = pd.DataFrame(pca.fit_transform(X_train))
    xs_test = pd.DataFrame(pca.transform(X_test))
    exp_var_pca = pca.explained_variance_ratio_
    print("Cumulative explained variance by truncated SVD:", np.cumsum(exp_var_pca))

    print(
        f"Final dataset shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
    )

    exp_var_pca = pca.explained_variance_ratio_
    print("Explained variance by each component:", exp_var_pca)
    print("Cumulative explained variance by truncated SVD:", np.cumsum(exp_var_pca)[-1])

    # Calculate the number of qubits needed for our new dimensionality
    # N_QUBITS = int(np.ceil(np.log2(N_COMPONENTS)))
    print(
        f"\nNumber of PCA components: {N_COMPONENTS} -> Number of Qubits needed: {N_QUBITS}"
    )

    # --- Step 2: Apply Quantum-SMOTE to the preprocessed training data ---

    # --- Apply Quantum-SMOTE ---
    if oversampling == "QSMOTE":
        print("\nClass distribution before Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "before qsmote")
        xs_train, y_train = quantum_smote(
            features=xs_train,
            labels=y_train.to_numpy(),
            n_qubits=N_QUBITS,
            desired_ratio=1.0,
            split_factor=1.0,
        )
        print("\nClass distribution after Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after QSMOTE")
    if oversampling == "SMOTE":
        smote = SMOTE(random_state=42)
        xs_train, y_train = smote.fit_resample(xs_train, y_train)
        visualize_data(xs_train, y_train, "after SMOTE")
    if oversampling == "NONE":
        print("---No oversampling was performed---")

if dataset == "breast_cancer":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = data.data
    Y = data.target
    # show type of X and Y
    if FIXING is True:
        X, _, Y, _ = train_test_split(
            X, Y, train_size=percent, stratify=Y, random_state=42
        )

    # === Step 7: Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y, random_state=42
    )

    # === Step 8 (Optional): Standard normalization ===
    # scaler = StandardScaler()
    # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
    # X_test = pd.DataFrame(scaler.fit_transform(X_train), columns=X_test.columns)
    # === Step 9 (Optional): Min-Max scaling ===
    min_max_scaler = MinMaxScaler(feature_range=(0, np.pi))
    # show datatype of X_train and X_test
    print(f"X_train type: {type(X_train)}, X_test type: {type(X_test)}")
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    N_COMPONENTS = N_QUBITS
    pca = PCA(n_components=N_QUBITS)
    xs_train = pd.DataFrame(pca.fit_transform(X_train))
    xs_test = pd.DataFrame(pca.transform(X_test))
    exp_var_pca = pca.explained_variance_ratio_
    print("Cumulative explained variance by PCA:", np.cumsum(exp_var_pca))

    print(
        f"Final dataset shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
    )

    exp_var_pca = pca.explained_variance_ratio_
    print("Explained variance by each component:", exp_var_pca)
    print("Cumulative explained variance by PCA:", np.cumsum(exp_var_pca)[-1])

    # Calculate the number of qubits needed for our new dimensionality
    # N_QUBITS = int(np.ceil(np.log2(N_COMPONENTS)))
    print(
        f"\nNumber of PCA components: {N_COMPONENTS} -> Number of Qubits needed: {N_QUBITS}"
    )

    # --- Step 2: Apply Quantum-SMOTE to the preprocessed training data ---

    # --- Apply Quantum-SMOTE ---
    if oversampling == "QSMOTE":
        print("\nClass distribution before Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "before QSMOTE")
        xs_train, y_train = quantum_smote(
            features=xs_train,
            labels=y_train,
            n_qubits=N_QUBITS,
            desired_ratio=1.0,
            split_factor=1.0,
        )
        visualize_data(xs_train, y_train, "after QSMOTE")
        print("\nClass distribution before Q-SMOTE:", Counter(y_train))
    if oversampling == "SMOTE":
        visualize_data(xs_train, y_train, "before SMOTE")
        smote = SMOTE(random_state=42)
        xs_train, y_train = smote.fit_resample(xs_train, y_train)
        visualize_data(xs_train, y_train, "after SMOTE")
    if oversampling == "NONE":
        print("---No oversampling was performed---")

if dataset == "immuno":
    data = pd.read_csv(DATA_DIR / "immuno.csv")

    # === Step 2: Report missing values ===
    print("\n--- Missing values BEFORE processing ---")
    print(data.isna().sum())

    # === Step 3: Impute missing values ===
    imputer_median = SimpleImputer(strategy="median")

    # Apply median imputation only to numeric columns
    for col in ["TIL_density", "Fraction_CD8"]:
        data[col] = imputer_median.fit_transform(data[[col]])

    # === Step 4: Remove duplicates and drop ID column ===
    print("Duplicate rows:", data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    data.drop(columns=["id"], inplace=True)

    # === Step 5: Encode categorical variables ===
    le = LabelEncoder()
    categorical_cols = [
        "gender",
        "med_history",
        "chronic_disease",
        "TCRA_neg",
        "TLS_type",
        "cancer_type",
        "CPI_status",
    ]

    for col in categorical_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    # === Step 6: Separate features and target ===
    X = data.drop(columns=["indication"])
    Y = data["indication"]

    print(f"\nFull dataset shape: {X.shape}, Target shape: {Y.shape}")

    if FIXING is True:
        X, _, Y, _ = train_test_split(
            X, Y, train_size=percent, stratify=Y, random_state=42
        )
        print(f"Dataset skimmed to {percent*100:.0f}% of original")

    # === Step 7: Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y, random_state=42
    )
    print(
        f"\nShapes -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}"
    )

    # === Step 8: Scaling (MinMax to [0, π]) ===
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # === Step 9: Dimensionality reduction (TruncatedSVD) ===
    N_COMPONENTS = N_QUBITS
    pca = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    xs_train = pd.DataFrame(pca.fit_transform(X_train))
    xs_test = pd.DataFrame(pca.transform(X_test))

    exp_var = pca.explained_variance_ratio_
    print("\nExplained variance per component:", exp_var)
    print("Cumulative explained variance by truncated SVD:", np.cumsum(exp_var)[-1])

    print(f"\nNumber of PCA components: {N_COMPONENTS} -> Number of Qubits: {N_QUBITS}")

    # === Step 10: Oversampling (if specified) ===
    if oversampling == "QSMOTE":
        print("\nClass distribution before Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "before QSMOTE")

        xs_train, y_train = quantum_smote(
            features=xs_train,
            labels=y_train.to_numpy(),
            n_qubits=N_QUBITS,
            desired_ratio=1.0,
            split_factor=1.0,
        )

        print("Class distribution after Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after QSMOTE")

    elif oversampling == "SMOTE":
        print("\nClass distribution before SMOTE:", Counter(y_train))
        smote = SMOTE(random_state=42)
        xs_train, y_train = smote.fit_resample(xs_train, y_train)
        print("Class distribution after SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after SMOTE")

    else:
        print("\n--- No oversampling performed ---")

    # === Step 11: Diagnostics (missing values after imputation) ===
    print("\n--- Missing values AFTER imputation ---")
    print(data.isna().sum())
    print("\n--- Missing values in PCA-transformed train set ---")
    print(pd.DataFrame(xs_train).isna().sum().sum(), "total missing values remaining")

if dataset == "signal":

    data = pd.read_csv(DATA_DIR / "Signalling_solid_cancer.csv")
    # Drop completely empty columns
    data = data.dropna(axis=1, how="all")

    # Clean column names
    data.columns = (
        data.columns.str.strip()
        .str.replace("–", "-", regex=False)
        .str.replace("−", "-", regex=False)
    )
    # === Step 2: Missing values report ===
    print("\n--- Missing values BEFORE preprocessing ---")
    print(data.isna().sum())

    # === Step 3: Impute missing numeric values (median) ===
    imputer_median = SimpleImputer(strategy="median")
    numeric_cols = ["Caveolin-1_count", "CDT1-R-V", "p38-MAPK-_pT180_Y182-R-V"]

    for col in numeric_cols:
        data[col] = imputer_median.fit_transform(data[[col]])

    # === Step 4: Remove duplicates & drop ID column ===
    print("Duplicate rows:", data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    data.drop(columns=["id"], inplace=True)

    # === Step 5: Encode categorical columns ===
    le = LabelEncoder()
    categorical_cols = [
        "gender",
        "Cyclin-E1_mut",
        "ATM_pS1981-mut",
        "PD-L1_expr",
        "Bcl2-R-C_alteration",
        "PRAS40_pT246-mut",
    ]
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col].astype(str))

    # === Step 6: Split features & target ===
    X = data.drop(columns=["SignificantAlt"])
    Y = data["SignificantAlt"]

    print(f"\nFull dataset shape: {X.shape}, Target shape: {Y.shape}")

    if FIXING:
        X, _, Y, _ = train_test_split(
            X, Y, train_size=percent, stratify=Y, random_state=42
        )
        print(f"Dataset skimmed to {percent*100:.0f}% of original")

    # === Step 7: Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y, random_state=42
    )

    # === Step 8: Scaling (Min–Max to [0, π]) ===
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # === Step 9: Dimensionality reduction (optional) ===
    N_COMPONENTS = N_QUBITS
    pca = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    xs_train = pd.DataFrame(pca.fit_transform(X_train))
    xs_test = pd.DataFrame(pca.transform(X_test))

    exp_var = pca.explained_variance_ratio_
    print("\nExplained variance per component:", exp_var)
    print("Cumulative explained variance by truncated SVD:", np.cumsum(exp_var)[-1])

    print(f"\nNumber of PCA components: {N_COMPONENTS} -> Number of Qubits: {N_QUBITS}")

    # === Step 10: Oversampling (optional) ===
    if oversampling == "QSMOTE":
        print("\nClass distribution before Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "before QSMOTE")

        xs_train, y_train = quantum_smote(
            features=xs_train,
            labels=y_train.to_numpy(),
            n_qubits=N_QUBITS,
            desired_ratio=1.0,
            split_factor=1.0,
        )

        print("Class distribution after Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after QSMOTE")

    elif oversampling == "SMOTE":
        print("\nClass distribution before SMOTE:", Counter(y_train))
        smote = SMOTE(random_state=42)
        xs_train, y_train = smote.fit_resample(xs_train, y_train)
        print("Class distribution after SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after SMOTE")

    else:
        print("\n--- No oversampling performed ---")

    # === Step 11: Missing-value diagnostics after preprocessing ===
    print("\n--- Missing values AFTER preprocessing ---")
    print(data.isna().sum())
    print("\n--- Missing values in PCA-transformed train set ---")
    print(pd.DataFrame(xs_train).isna().sum().sum(), "total missing values remaining")

if dataset == "coldmss":

    data = pd.read_csv(DATA_DIR / "coldmss.csv")

    # Drop empty columns (e.g., from trailing commas)
    data = data.dropna(axis=1, how="all")

    # Clean column names
    data.columns = (
        data.columns.str.strip()
        .str.replace("–", "-", regex=False)
        .str.replace("−", "-", regex=False)
    )

    print("\n--- Missing values BEFORE preprocessing ---")
    print(data.isna().sum())

    # Drop ID early so it never gets encoded/scaled
    if "id" in data.columns:
        data.drop(columns=["id"], inplace=True)

    # Split features/target BEFORE encoding so we never touch y by accident
    X = data.drop(columns=["response"])
    Y = data["response"]

    print(f"\nFull dataset shape: {X.shape}, Target shape: {Y.shape}")
    print("Duplicate rows:", pd.concat([X, Y], axis=1).duplicated().sum())

    # --- Encode categorical columns automatically on X only ---
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Categorical columns detected: {cat_cols}")
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Optional skim
    if FIXING:
        X, _, Y, _ = train_test_split(
            X, Y, train_size=percent, stratify=Y, random_state=42
        )
        print(f"Dataset skimmed to {percent*100:.0f}% of original")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y, random_state=42
    )
    print(
        f"\nShapes -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}"
    )

    # Sanity check: everything numeric before scaling
    print("\n--- Dtypes before scaling (should all be numeric) ---")
    print(X_train.dtypes)

    # Min–Max scaling to [0, π]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Dimensionality reduction (PCA to N_QUBITS)
    N_COMPONENTS = N_QUBITS
    pca = TruncatedSVD(n_components=N_QUBITS, random_state=42)
    xs_train = pd.DataFrame(pca.fit_transform(X_train))
    xs_test = pd.DataFrame(pca.transform(X_test))

    exp_var = pca.explained_variance_ratio_
    print("\nExplained variance per component:", exp_var)
    print("Cumulative explained variance by truncated SVD:", np.cumsum(exp_var)[-1])
    print(f"\nNumber of PCA components: {N_COMPONENTS} -> Number of Qubits: {N_QUBITS}")

    # Oversampling
    if oversampling == "QSMOTE":
        print("\nClass distribution before Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "before QSMOTE")
        xs_train, y_train = quantum_smote(
            features=xs_train,
            labels=y_train.to_numpy(),
            n_qubits=N_QUBITS,
            desired_ratio=1.0,
            split_factor=1.0,
        )
        print("Class distribution after Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after QSMOTE")

    elif oversampling == "SMOTE":
        print("\nClass distribution before SMOTE:", Counter(y_train))
        smote = SMOTE(random_state=42)
        xs_train, y_train = smote.fit_resample(xs_train, y_train)
        print("Class distribution after SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after SMOTE")

    else:
        print("\n--- No oversampling performed ---")

    # Diagnostics
    print("\n--- Missing values AFTER preprocessing ---")
    print(pd.concat([X, Y], axis=1).isna().sum())
    print("\n--- Missing values in PCA-transformed train set ---")
    print(pd.DataFrame(xs_train).isna().sum().sum(), "total missing values remaining")
    # reset index of xs_train and xs_test
    xs_train.reset_index(drop=True, inplace=True)
    xs_test.reset_index(drop=True, inplace=True)
if dataset == "crc_lm_cht":
    # === Step 1: Load and clean data ===
    data = pd.read_csv(DATA_DIR / "CRC_LM_CHT.csv")

    # Show any empty rows (all NaN)
    empty_rows = data[data.isna().all(axis=1)]
    print(f"Empty rows: {len(empty_rows)}")
    # Drop empty columns and clean column names
    empty_rows = data[data.isna().all(axis=1)]
    print(f"Dropping {len(empty_rows)} completely empty rows.")
    data = data.dropna(how="all").reset_index(drop=True)

    empty_rows = data[data.isna().all(axis=1)]
    print(f"Empty rows: {len(empty_rows)}")

    data = data.dropna(axis=1, how="all")
    data.columns = (
        data.columns.str.strip()
        .str.replace("–", "-", regex=False)
        .str.replace("−", "-", regex=False)
    )

    print("\n--- Missing values BEFORE preprocessing ---")
    print(data.isna().sum())

    # === Step 2: Drop ID column ===
    if "id" in data.columns:
        data.drop(columns=["id"], inplace=True)
        print("Dropped ID column.")

    # === Step 3: Handle numeric commas and placeholders ===
    # Convert European comma decimals (e.g., "36,6") and replace "N/A" with NaN
    data = data.replace({"N/A": np.nan, "n/a": np.nan, "NA": np.nan})
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].str.replace(",", ".", regex=False)

    # Coerce only the numeric-LOOKING columns. Coercing unconditionally turned the
    # five genuinely categorical columns (gender, Obesity, localisation,
    # cytochrome_class, smoking_status) into all-NaN float columns, which
    # SimpleImputer then silently dropped -- raising "Columns must be same length
    # as key" at the imputation step below. Columns that do not survive coercion
    # are left as objects and label-encoded in Step 7, as in the other branches.
    for col in data.columns:
        coerced = pd.to_numeric(data[col], errors="coerce")
        if not coerced.isna().all():
            data[col] = coerced

    # === Step 4: Drop duplicates ===
    print("Duplicate rows:", data.duplicated().sum())
    data.drop_duplicates(inplace=True)

    # === Step 5: Separate features and target ===
    X = data.drop(columns=["response"])
    Y = data["response"]

    print(f"\nFull dataset shape: {X.shape}, Target shape: {Y.shape}")

    # === Step 6: Impute numeric columns (median) ===
    imputer_median = SimpleImputer(strategy="median")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X[num_cols] = imputer_median.fit_transform(X[num_cols])

    # === Step 7: Encode categorical columns automatically ===
    le = LabelEncoder()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Categorical columns detected: {cat_cols}")
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # === Step 8: Optional data skim (for smaller runs) ===
    if FIXING:
        X, _, Y, _ = train_test_split(
            X, Y, train_size=percent, stratify=Y, random_state=42
        )
        print(f"Dataset skimmed to {percent*100:.0f}% of original")

    # === Step 9: Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y, random_state=42
    )
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    print(
        f"\nShapes -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}"
    )

    # Sanity check: all numeric before scaling
    print("\n--- Dtypes before scaling (should all be numeric) ---")
    print(X_train.dtypes)

    # === Step 10: Scaling (Min–Max to [0, π]) ===
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # === Step 11: Dimensionality reduction (PCA) ===
    N_COMPONENTS = N_QUBITS
    pca = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    xs_train = pd.DataFrame(pca.fit_transform(X_train))
    xs_test = pd.DataFrame(pca.transform(X_test))

    exp_var = pca.explained_variance_ratio_
    print("\nExplained variance per component:", exp_var)
    print("Cumulative explained variance by truncated SVD:", np.cumsum(exp_var)[-1])
    print(f"\nNumber of PCA components: {N_COMPONENTS} -> Number of Qubits: {N_QUBITS}")

    # === Step 12: Oversampling (optional) ===
    if oversampling == "QSMOTE":
        print("\nClass distribution before Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "before QSMOTE")

        xs_train, y_train = quantum_smote(
            features=xs_train,
            labels=y_train.to_numpy(),
            n_qubits=N_QUBITS,
            desired_ratio=1.0,
            split_factor=1.0,
        )

        print("Class distribution after Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after QSMOTE")

    elif oversampling == "SMOTE":
        print("\nClass distribution before SMOTE:", Counter(y_train))
        smote = SMOTE(random_state=42)
        xs_train, y_train = smote.fit_resample(xs_train, y_train)
        print("Class distribution after SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after SMOTE")

    else:
        print("\n--- No oversampling performed ---")

    # === Step 13: Diagnostics ===
    print("\n--- Missing values AFTER preprocessing ---")
    print(pd.concat([X, Y], axis=1).isna().sum())
    print("\n--- Missing values in PCA-transformed train set ---")
    print(pd.DataFrame(xs_train).isna().sum().sum(), "total missing values remaining")

if dataset == "pdac":
    data = pd.read_csv(DATA_DIR / "PDAC.csv", na_values=["N/A", "NA", ""])
    # Drop completely empty rows (if any)
    empty_rows = data[data.isna().all(axis=1)]
    if len(empty_rows) > 0:
        print(f"🧹 Dropping {len(empty_rows)} completely empty rows.")
    data = data.dropna(how="all").reset_index(drop=True)

    # Clean column names
    data.columns = (
        data.columns.str.strip()
        .str.replace("–", "-", regex=False)
        .str.replace("−", "-", regex=False)
    )

    print("\n--- Missing values BEFORE preprocessing ---")
    print(data.isna().sum())

    # === Step 2: Drop ID column ===
    if "id" in data.columns:
        data.drop(columns=["id"], inplace=True)
        print("🗑️  Dropped ID column.")

    # === Step 3: Remove duplicate rows ===
    print("Duplicate rows:", data.duplicated().sum())
    data.drop_duplicates(inplace=True)

    # === Step 4: Separate features and target ===
    if "prognosis" not in data.columns:
        raise KeyError("❌ Target column 'prognosis' not found in dataset.")

    X = data.drop(columns=["prognosis"])
    Y = data["prognosis"]

    # Drop rows with missing target values (shouldn't happen but safe)
    missing_target = Y.isna().sum()
    if missing_target > 0:
        print(f"⚠️ Dropping {missing_target} samples with missing target labels.")
        X = X.loc[~Y.isna()].reset_index(drop=True)
        Y = Y.dropna().reset_index(drop=True)

    print(f"\nFull dataset shape: {X.shape}, Target shape: {Y.shape}")

    # === Step 5: Impute missing numeric values (median) ===
    imputer_median = SimpleImputer(strategy="median")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        X[num_cols] = imputer_median.fit_transform(X[num_cols])

    # === Step 6: Encode categorical columns automatically ===
    le = LabelEncoder()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Categorical columns detected: {cat_cols}")
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # === Step 7: Optional data skim ===
    if FIXING:
        X, _, Y, _ = train_test_split(
            X, Y, train_size=percent, stratify=Y, random_state=42
        )
        print(f"Dataset skimmed to {percent*100:.0f}% of original")

    # === Step 8: Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=Y, random_state=42
    )
    print(
        f"\nShapes -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}"
    )

    # Sanity check: all numeric before scaling
    print("\n--- Dtypes before scaling (should all be numeric) ---")
    print(X_train.dtypes)

    # === Step 9: Scaling (Min–Max to [0, π]) ===
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # === Step 10: Dimensionality reduction (PCA) ===
    N_COMPONENTS = N_QUBITS
    pca = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    xs_train = pd.DataFrame(pca.fit_transform(X_train))
    xs_test = pd.DataFrame(pca.transform(X_test))

    exp_var = pca.explained_variance_ratio_
    print("\nExplained variance per component:", exp_var)
    print("Cumulative explained variance by truncated SVD:", np.cumsum(exp_var)[-1])
    print(f"\nNumber of PCA components: {N_COMPONENTS} -> Number of Qubits: {N_QUBITS}")

    # === Step 11: Oversampling (optional) ===
    if oversampling == "QSMOTE":
        print("\nClass distribution before Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "before QSMOTE")
        xs_train, y_train = quantum_smote(
            features=xs_train,
            labels=y_train.to_numpy(),
            n_qubits=N_QUBITS,
            desired_ratio=1.0,
            split_factor=1.0,
        )
        print("Class distribution after Q-SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after QSMOTE")

    elif oversampling == "SMOTE":
        print("\nClass distribution before SMOTE:", Counter(y_train))
        smote = SMOTE(random_state=42)
        xs_train, y_train = smote.fit_resample(xs_train, y_train)
        print("Class distribution after SMOTE:", Counter(y_train))
        visualize_data(xs_train, y_train, "after SMOTE")

    else:
        print("\n--- No oversampling performed ---")

    # === Step 12: Diagnostics ===
    print("\n--- Missing values AFTER preprocessing ---")
    print(pd.concat([X, Y], axis=1).isna().sum())
    print("\n--- Missing values in PCA-transformed train set ---")
    print(pd.DataFrame(xs_train).isna().sum().sum(), "total missing values remaining")

if dataset == "stroke" and oversampling != "QSMOTE":
    # Convert y_train and y_test to numpy arrays
    y_train = y_train.to_numpy(dtype=int)
    y_test = y_test.to_numpy(dtype=int)

# Count classes
train_counts = Counter(y_train)
test_counts = Counter(y_test)


# Convert to proportions
def proportions(sums):
    """Calculate proportions"""
    total = sum(sums.values())
    return {cls: count / total for cls, count in sums.items()}


train_props = proportions(train_counts)
test_props = proportions(test_counts)
from sklearn.metrics import make_scorer, matthews_corrcoef

N_train = xs_train.shape[0]
GRID_SEARCH = True
# scoring_parameter = ['roc_auc', 'f1']
scoring_parameter = {
    "auc": "roc_auc",
    "mcc": make_scorer(matthews_corrcoef),
}
refit_parameter = "mcc"
C = 1
C_values = [0.001, 0.01, 0.1, 1, 5, 10, 100, 200, 1000]
gamma_values = [0.001, 0.01, 0.1, 1, 5, 10, 100, 200, 1000, 10000]
param_grid = {
    "C": C_values,
}
param_grid_rbf = {
    "C": C_values,
    "gamma": gamma_values,
}
# Seems like gpu doesnt bring too much benefit for these circuits
# Device Setup..
dev = qml.device("lightning.qubit", wires=N_QUBITS, batch_obs=True, shots=1000)


def plot_kernel_matrix(
    K,
    labels,
    title="Quantum Kernel Matrix",
    filename=None,
    path=None,
    save_fig=True,
    show_colorbar=True,
    cmap="Blues",
    vmin=None,
    vmax=None,
    add_class_separators=True,
    annotate=False,
    box_size_inches=0.3,
    set_ticks=False,
):
    """
    Plot a kernel matrix with consistent box sizes regardless of matrix dimensions.

    Args:
        K (numpy.ndarray): The kernel matrix to plot
        labels (numpy.ndarray): Class labels for each sample
        ...
        box_size_inches (float): Size of each matrix cell in inches (controls consistency)
    """
    # Sort by class labels for better visualization
    sorted_idx = np.argsort(labels)
    K_sorted = K[sorted_idx][:, sorted_idx]
    labels_sorted = labels[sorted_idx]

    # Calculate figure size based on matrix dimensions to maintain consistent box size
    n_samples = K.shape[0]
    matrix_size_inches = n_samples * box_size_inches
    colorbar_space = 1.0 if show_colorbar else 0.0

    # Create figure with size proportional to matrix dimensions
    fig_width = matrix_size_inches + colorbar_space
    fig_height = matrix_size_inches
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # Plot the heatmap
    heatmap = sns.heatmap(
        K_sorted,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=show_colorbar,
        xticklabels=set_ticks,
        yticklabels=set_ticks,
        ax=ax,
        # Use a symmetric colormap if kernel values range from -1 to 1
        norm=colors.Normalize(
            vmin=vmin if vmin is not None else K.min(),
            vmax=vmax if vmax is not None else K.max(),
        ),
    )

    # Add class boundary lines if requested
    if add_class_separators:
        # Find the indices where class labels change
        class_boundaries = [0]
        for i in range(1, len(labels_sorted)):
            if labels_sorted[i] != labels_sorted[i - 1]:
                class_boundaries.append(i)
        class_boundaries.append(len(labels_sorted))

        # Add lines at class boundaries
        for boundary in class_boundaries:
            if boundary > 0 and boundary < len(labels_sorted):
                ax.axhline(
                    y=boundary, color="black", linestyle="--", linewidth=1.5, alpha=0.7
                )
                ax.axvline(
                    x=boundary, color="black", linestyle="--", linewidth=1.5, alpha=0.7
                )

    # Add value annotations if requested
    if annotate and K.shape[0] <= 20:  # Only annotate for small matrices
        for i in range(K_sorted.shape[0]):
            for j in range(K_sorted.shape[1]):
                text_color = "white" if K_sorted[i, j] > 0.7 else "black"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{K_sorted[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

    # Add title and labels
    if title:
        plt.title(title, fontsize=14, fontweight="bold")

    # Add class labels to tick marks (only if set_ticks=True)
    if set_ticks and len(np.unique(labels)) <= 10:
        unique_labels = np.unique(labels_sorted)
        class_centers = {}

        # Calculate center position for each class
        for label in unique_labels:
            positions = np.where(labels_sorted == label)[0]
            class_centers[label] = (positions[0] + positions[-1]) / 2

        # Set tick positions and labels on both axes
        tick_positions = list(class_centers.values())
        tick_labels = [f"Class {int(label)}" for label in class_centers.keys()]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, rotation=0)
    else:
        # Remove all ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()

    # Save or show the plot
    if filename and path and save_fig == True:
        plt.savefig(os.path.join(path, filename), dpi=450, bbox_inches="tight")
        plt.close()
        print(f"------Saved figure to {os.path.join(path, filename)}------")
    else:
        plt.show()
        print("------Nothing was saved!!------")

    return fig


def block_average_with_labels(K, labels, block_size):
    """
    Block-average a kernel matrix and aggregate labels by majority in each block.
    Args:
        K (np.ndarray): Square kernel matrix [n_samples, n_samples]
        labels (np.ndarray): Label vector [n_samples]
        block_size (int): Size of each block for averaging
    Returns:
        K_agg (np.ndarray): Block-averaged kernel matrix
        y_agg (np.ndarray): Aggregated labels (majority per block)
    """
    n = K.shape[0]
    m = (n // block_size) * block_size
    # Truncate to fit complete blocks
    K = K[:m, :m]
    labels = labels[:m]
    # Block-averaged kernel
    K_agg = K.reshape(m // block_size, block_size, m // block_size, block_size).mean(
        axis=(1, 3)
    )
    np.fill_diagonal(K_agg, 1.0)
    # Aggregated labels: majority in each block
    y_agg = np.array(
        [
            np.bincount(labels[i * block_size : (i + 1) * block_size]).argmax()
            for i in range(m // block_size)
        ]
    )
    return K_agg, y_agg


def get_optimal_block_size(xs_test, target_blocks=20):
    """Calculate block size to achieve a target number of blocks."""
    n_samples = len(xs_test)
    block_size = max(1, n_samples // target_blocks)
    return block_size


def kernel_entry_param(i, j, A, B, params):
    """Compute single kernel entry with parameters"""
    # Create a copy of params with TROTT_STEPS as integer
    circuit_params = [params[0], int(params[1]), int(params[2])]
    return i, j, kernel_circ(A[i], B[j], circuit_params)[0]


def compute_kernel_param(A, B, params, n_jobs=jobs):
    """
    Compute kernel matrix with parameters

    Args:
        A: First dataset of shape (n_samples_A, n_features)
        B: Second dataset of shape (n_samples_B, n_features)
        params: Kernel parameters to pass to kernel_circ
        n_jobs: Number of parallel jobs (-1 for all available)
    """
    N = len(A)
    M = len(B)
    K = np.zeros((N, M))

    if np.array_equal(A, B):
        pairs = [(i, j) for i in range(N) for j in range(i, N)]
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(kernel_entry_param)(i, j, A, B, params)
            for i, j in tqdm(pairs, desc="Symmetric train kernel", disable=align)
        )
        for i, j, val in results:
            K[i, j] = val
            K[j, i] = val
    else:
        pairs = [(i, j) for i in range(N) for j in range(M)]
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(kernel_entry_param)(i, j, A, B, params)
            for i, j in tqdm(pairs, desc="Test kernel", disable=align)
        )
        for i, j, val in results:
            K[i, j] = val

    return K


# --------------------------
# -----CLASSICAL SVM--------
# --------------------------
if oversampling in ("QSMOTE", "SMOTE"):
    classweight = None
else:
    classweight = "balanced"

classicsvm = SVC(
    kernel="rbf",
    class_weight=classweight,
    C=C,
    probability=True,
    random_state=42,  # optional but recommended for reproducibility
).fit(xs_train, y_train)

y_pred = classicsvm.predict(xs_test)

# --- ADD THESE DIAGNOSTIC LINES ---
print("\n--- DIAGNOSTICS ---")
# Use original numpy to be safe
import numpy as original_numpy

unique_y_test = original_numpy.unique(y_test)
unique_y_pred = original_numpy.unique(y_pred)

print(
    f"Type of y_test: {type(y_test.iloc[0]) if hasattr(y_test, 'iloc') else type(y_test[0])}"
)
print(f"Type of y_pred: {type(y_pred[0])}")
print(f"Unique values in y_test: {unique_y_test}")
print(f"Unique values in y_pred: {unique_y_pred}")
print("---------------------\n")
# --- END OF DIAGNOSTIC LINES ---

# Now print your report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(
    f"Classical SVM (BP):\n %s"
    % classification_report(y_test, classicsvm.predict(xs_test))
)
print(
    "ROC AUC Score:",
    roc_auc_score(y_test, classicsvm.predict(xs_test), multi_class=roc_multiclass),
)
# GRID Search
if GRID_SEARCH is True:
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=42)
    grid = GridSearchCV(
        estimator=classicsvm,
        param_grid=param_grid_rbf,
        refit=refit_parameter,
        n_jobs=jobs,
        cv=cv,
        scoring=scoring_parameter,
    )
    grid_result = grid.fit(xs_train, y_train)
    print(
        f"Best {scoring_parameter}: %f using %s"
        % (grid_result.best_score_, grid_result.best_params_)
    )
    # report all configurations
best_svm_class = grid.best_estimator_
print(confusion_matrix(y_test, best_svm_class.predict(xs_test)))
print(
    f"Classical SVM (GS acting on {scoring_parameter}):\n %s"
    % classification_report(y_test, best_svm_class.predict(xs_test))
)
y_score_classical = best_svm_class.decision_function(xs_test)
###
### These are then used for all following ROC-Images
###

fpr_c, tpr_c, _ = roc_curve(y_test, y_score_classical)
auc_c = auc(fpr_c, tpr_c)

# If AUC < 0.5 → invert the scores
if auc_c < 0.5:
    print(f"⚠️ Classical AUC < 0.5 ({auc_c:.3f}) → inverting decision scores")
    y_score_classical = -y_score_classical
    fpr_c, tpr_c, _ = roc_curve(y_test, y_score_classical)
    auc_c = auc(fpr_c, tpr_c)  # new corrected AUC

print("Classical ROC AUC:", auc_c)

# NOTE: the legacy AngleEmbedding ('rotation') kernel block that used to sit
# here was disabled as a string literal and has been removed for release.
# --------------------------
# -----IsingXYZ-Emb--------
# --------------------------
# add hamilton encoding (https://arxiv.org/pdf/2310.11891)
# from pennylane import numpy as np
dev = qml.device("lightning.qubit", wires=N_QUBITS, batch_obs=True, shots=3000)
EV_TIME = 0.1
TROTT_STEPS = 20
N_LAYERS = 3
# params = np.array([EV_TIME, TROTT_STEPS], requires_grad=True)  # Add requires_grad=True
params = [EV_TIME, TROTT_STEPS, N_LAYERS]  # Add requires_grad=True

model_name = "ising_emb" if TOPOLOGY == "chain" else f"ising_emb_{TOPOLOGY}"


def Hamiltonian_feature_map(x, params, wires, initial_state=True, seed=42):
    """
    Combined quantum feature map with Haar-like initial state and Hamiltonian evolution.

    Args:
        x: Input data point
        params: [t, T] where t is evolution time and T is Trotter steps
        wires: Wires to apply the circuit on
        initial_state: Whether to apply the initial Haar-like state (set False for adjoint)
        seed: Random seed for reproducibility
    """
    n_qubits = len(wires)
    t, T, N_LAYERS = params  # Fixed parameter order

    # Apply random initial state if requested
    if initial_state:
        rng = np.random.default_rng(seed)
        weights = rng.normal(loc=0, scale=1, size=(N_LAYERS, n_qubits, 3))
        qml.StronglyEntanglingLayers(weights, wires=wires)

    # Apply Hamiltonian feature map. coupling_pairs is the single definition of
    # which wire pair carries which feature -- shared with kernel_align's fast
    # path and verify_fast_kernel, so the two can never drift apart again.
    pairs = kernel_align.coupling_pairs(n_qubits, len(x), TOPOLOGY)
    for _ in range(T):  # Trotter steps
        for j, (wa, wb) in enumerate(pairs):
            theta = (t / T) * x[j]
            qml.IsingXX(2 * theta, wires=[wires[wa], wires[wb]])
            qml.IsingYY(2 * theta, wires=[wires[wa], wires[wb]])
            qml.IsingZZ(2 * theta, wires=[wires[wa], wires[wb]])


@qml.qnode(dev)
def kernel_circ(a, b, params):
    """
    Parameterized kernel circuit using the consolidated feature map

    Args:
        a: First data point
        b: Second data point
        params: [t, T] where t is evolution time and T is Trotter steps
    """
    wires = range(N_QUBITS)

    # Apply feature map to first data point
    Hamiltonian_feature_map(a, params, wires=wires, initial_state=True)

    # Apply adjoint feature map to second data point
    # Note: For the adjoint, we skip applying the initial state again
    qml.adjoint(Hamiltonian_feature_map)(b, params, wires=wires, initial_state=True)

    return qml.probs(wires=wires)


# Redirect every Gram-matrix call site at once. Python resolves globals at call
# time, so rebinding the name is enough: compute_kernel_param(A, B, params=...)
# below now goes through the feature-state path. Verified equivalent to
# kernel_circ to ~1e-15 by verify_fast_kernel.py.
if KERNEL_MODE == "exact":
    compute_kernel_param = kernel_align.make_exact_kernel_fn(
        N_QUBITS, seed=42, topology=TOPOLOGY
    )


def evaluate_kernel(params, model_name, target_blocks=40, prefix="", psep="_"):
    """Gram -> heatmap -> C search -> 4-fold CV ROC -> test ROC with bootstrap CI.

    Run once against the unaligned feature-map parameters and once against the
    aligned ones; comparing the two calls is how the "does alignment help?"
    result is produced. `prefix` and `psep` only reproduce the two output-file
    naming conventions already used in ~/results, and change nothing numeric.
    """
    stem = f"OS{oversampling}_{N_QUBITS}qubits_{percent}percent.png"

    # --- Train Gram + heatmap -------------------------------------------------
    filename = f"{prefix}{dataset}_{model_name}_params{params}_kernel_{stem}"
    block_size = get_optimal_block_size(xs_train, target_blocks=target_blocks)
    K_full = compute_kernel_param(xs_train, xs_train, params=params)  # [n, n]
    K_agg, y_agg = block_average_with_labels(K_full, y_train, block_size=block_size)
    plot_kernel_matrix(
        K_agg,
        y_agg,
        title="Quantum Kernel Matrix (Train Set)",
        filename=filename,
        path=folder_kernel,
        save_fig=save_fig,
    )

    # --- C search on precomputed folds ---------------------------------------
    cv_inner = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=42)

    # Cache per-fold kernel slices once (avoids re-slicing per C)
    cached_folds = []
    for train_idx, test_idx in cv_inner.split(xs_train, y_train):
        cached_folds.append(
            (
                K_full[np.ix_(train_idx, train_idx)],
                K_full[np.ix_(test_idx, train_idx)],
                y_train[train_idx],
                y_train[test_idx],
            )
        )

    # Mean/std AUC for a given C using decision_function (no probability calibration)
    def mean_auc_for_C(C: float) -> tuple[float, float]:
        clf = SVC(kernel="precomputed", C=C, class_weight="balanced", probability=False)
        aucs = []
        for K_tr, K_te, y_tr, y_te in cached_folds:
            clf.fit(K_tr, y_tr)
            aucs.append(roc_auc_score(y_te, clf.decision_function(K_te)))
        return float(np.mean(aucs)), float(np.std(aucs))

    def sweep(Cs):
        try:
            out = Parallel(n_jobs=jobs, verbose=0)(
                delayed(mean_auc_for_C)(C) for C in Cs
            )
        except Exception:
            out = [mean_auc_for_C(C) for C in Cs]
        return [(C, m, s) for C, (m, s) in zip(Cs, out)]

    coarse_results = sweep(param_grid.get("C", np.logspace(-4, 4, 13)))
    best_C_coarse = max(coarse_results, key=lambda t: t[1])[0]

    # Fine grid: +/-1 decade around the coarse best
    fine_results = sweep(
        np.logspace(np.log10(best_C_coarse) - 1, np.log10(best_C_coarse) + 1, 15)
    )
    best_C, best_auc, best_std = max(fine_results, key=lambda t: t[1])

    for label, results in (("Coarse", coarse_results), ("Fine", fine_results)):
        print(f"{label} results (C, mean AUC, std):")
        for C, m, s in results:
            print(f"  C={C:.6g}  mean={m:.4f}  std={s:.4f}")
    print(f"\n Best C: {best_C:.6g} with AUC = {best_auc:.4f} (+/-{best_std:.4f})")

    K_test_final = compute_kernel_param(xs_test, xs_train, params=params)

    # --- 4-fold CV ROC on the train set --------------------------------------
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    for train_idx, val_idx in cv.split(xs_train, y_train):
        clf = SVC(
            kernel="precomputed", C=best_C, probability=True, class_weight="balanced"
        )
        clf.fit(K_full[np.ix_(train_idx, train_idx)], y_train[train_idx])
        y_score = clf.decision_function(K_full[np.ix_(val_idx, train_idx)])

        fpr, tpr, _ = roc_curve(y_train[val_idx], y_score)
        aucs.append(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    filename = f"{prefix}{dataset}5fold_{model_name}_params{psep}{params}_{stem}"
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.figure(figsize=(7, 6))
    plt.plot(
        mean_fpr,
        mean_tpr,
        label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
        lw=2,
    )
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.2,
        label="± 1 std. dev.",
    )
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random 0.5")
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("1 - Specificity", **ROC_LABEL_FONT)
    plt.ylabel("Sensitivity", **ROC_LABEL_FONT)
    plt.legend(loc="lower right", frameon=False)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_roc, filename), dpi=300)
    plt.close()

    # --- Held-out test ROC, against the classical baseline --------------------
    best_model = SVC(
        kernel="precomputed", C=best_C, probability=True, class_weight="balanced"
    )
    best_model.fit(K_full, y_train)
    y_pred = best_model.predict(K_test_final)
    y_score = best_model.decision_function(K_test_final)

    print("\n HamiltonianFeatureMap-Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_score))

    filename = f"{prefix}{dataset}roccomp_{model_name}_params{psep}{params}_{stem}"
    fpr_q, tpr_q, _ = roc_curve(y_test, y_score)
    roc_auc_q = auc(fpr_q, tpr_q)

    mean_fpr, mean_tpr, std_tpr, boot_aucs = bootstrap_roc(y_test, y_score)
    avg_std, max_std = np.mean(std_tpr), np.max(std_tpr)
    auc_lo, auc_hi = np.percentile(boot_aucs, [2.5, 97.5])

    plt.figure(figsize=(6, 5))
    # Same seed as the quantum bootstrap, so both AUC spreads come from identical resamples.
    *_, boot_aucs_c = bootstrap_roc(y_test, y_score_classical)
    (line_q,) = plt.plot(fpr_q, tpr_q, linewidth=2)
    (line_c,) = plt.plot(fpr_c, tpr_c, linewidth=2, linestyle="--")
    (line_r,) = plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("1 - Specificity", **ROC_LABEL_FONT)
    plt.ylabel("Sensitivity", **ROC_LABEL_FONT)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    table_legend(
        plt.gca(),
        [line_q, line_c, line_r],
        ["Quantum Kernel SVM", "RBF Kernel SVM", "Random"],
        [
            f"{roc_auc_q:.3f} ± {np.std(boot_aucs):.3f}",
            f"{auc_c:.3f} ± {np.std(boot_aucs_c):.3f}",
            "0.5",
        ],
    )
    path = os.path.join(folder_roc, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    record_roc_comparison(
        path,
        roc_auc_q,
        auc_c,
        ci=(auc_lo, auc_hi),
        dataset=dataset,
        model=model_name,
        qubits=N_QUBITS,
    )


if GRID_SEARCH and not ALIGN_ONLY:
    evaluate_kernel(params, model_name)

if align is False:
    sys.exit("Alignment skipped, exiting...")
print("Starting alignment procedure...")

if ALIGN_PRESET is not None:
    opt_params = list(ALIGN_PRESET)
    print(f"Using preset alignment params: {opt_params}")
else:
    # N_LAYERS is held fixed rather than searched. It only selects a
    # data-independent initial state V_L|0>, so KTA(L) is an iid draw with no
    # structure and the old +/-1 hill-climb over L was selecting noise -- the
    # spread across L is typically larger than the whole gain from t and T.
    # L_values re-scores the winner across L and reports that spread instead.
    align_result = kernel_align.search_alignment(
        xs_train,
        y_train,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        topology=TOPOLOGY,
        t_transform="clip",
        L_values=[1, 2, 3, 4, 5],
        dataset=dataset,
        model_name=model_name,
        out_dir=ALIGN_OUT,
        save_fig=save_fig,
        verbose=True,
    )
    opt_params = align_result["opt_params"]

if ALIGN_ONLY:
    sys.exit(f"Alignment complete: params={opt_params}")

# redefine params as opt params
params = opt_params
model_name = f"{model_name}_aligned"
if GRID_SEARCH and not ALIGN_ONLY:
    evaluate_kernel(params, model_name, target_blocks=20, prefix="aligned", psep="")
