# Model: Linear Regression
# Test Size: 0.4
# Kernel: polynomial

import os
import sys
import numpy as np
import pandas as pd

def find_file(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    for base in [script_dir, os.path.dirname(script_dir)]:
        path = os.path.join(base, 'MachineLearningCSV', filename)
        if os.path.exists(path):
            return path
    return filename
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    candidate = os.path.join(script_dir, filename)
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(script_dir, '..', filename)
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(r'c:\b.tech\sem III\python\python class\MachineLearningCVE', filename)
    if os.path.exists(candidate):
        return candidate
    return filename

df1 = pd.read_csv(find_file('Tuesday-WorkingHours.pcap_ISCX.csv'), low_memory=True)
df2 = pd.read_csv(find_file('Wednesday-workingHours.pcap_ISCX.csv'), low_memory=True)
df3 = pd.read_csv(find_file('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'), low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=0,
    stratify=y
)

from sklearn.kernel_approximation import Nystroem
k_pca = Nystroem(n_components=2, kernel="poly", degree=3, gamma=15, coef0=1, random_state=0)

X_train = k_pca.fit_transform(X_train)
X_test = k_pca.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_cont = regressor.predict(X_test)

# Discretize continuous predictions to nearest valid class label for classification metrics
y_pred = np.clip(np.rint(y_pred_cont).astype(int), 0, len(np.unique(y_train)) - 1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

print("Regression Metrics:")
print("Mean Squared Error : {:.4f}".format(mean_squared_error(y_test, y_pred_cont)))
print("R2 Score           : {:.4f}".format(r2_score(y_test, y_pred_cont)))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n")
print(cm)

print("\nAccuracy  : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred, average='weighted', zero_division=0)))
print("Recall    : {:.4f}".format(recall_score(y_test, y_pred, average='weighted', zero_division=0)))
print("F1 Score  : {:.4f}".format(f1_score(y_test, y_pred, average='weighted', zero_division=0)))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

sys.stdout.flush()
os._exit(0)
