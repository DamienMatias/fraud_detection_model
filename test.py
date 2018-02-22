import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools
import scipy.stats as st
import seaborn as sns
from pylab import rcParams

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV


sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

df = pd.read_csv("creditcard.csv")

df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Time', 'Amount'], axis=1)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0, shuffle=False)
print(df_train)
print(len(df_test))

y_train = df_train['Class']
y_test = df_test['Class']

X_train = df_train.drop(['Class'], axis=1)
X_test = df_test.drop(['Class'], axis=1)

print(len(X_train))

# Number of data points in the minority class
print(len(df_train[df_train.Class == 0]))
number_records_fraud = len(y_train[y_train == 1])
print(number_records_fraud)
fraud_indices = np.array(y_train[y_train == 1].index)
print(fraud_indices)
print(max(fraud_indices))
# Picking the indices of the normal classes
normal_indices = df_train[df_train.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)
print(max(random_normal_indices))
# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
print(max(under_sample_indices))
print(len(df_train))
# Under sample dataset
under_sample_data = df_train.iloc[under_sample_indices,:]