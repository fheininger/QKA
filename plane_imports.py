"""Shared import block, star-imported by QSVM_template.py. No logic.

Trimmed to the names the driver actually references; anything a single module
needs on its own is imported there.
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             make_scorer, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
