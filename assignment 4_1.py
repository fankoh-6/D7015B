# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 07:57:11 2025

@author: fankoh-6
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:36:00 2025

@author: fankoh-6
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

#%% Load datasets into one dataframe
files = ["Trail1.csv", "Trail2.csv", "Trail3.csv"]

def load_and_combine_data(files):
    dataframes = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

df = load_and_combine_data(files)

print("Before preprocessing:")
print(df.head())
print(df['event'].unique())

#%% Drop the unnecessary data columns

def preprocess_data(df):
    
    df.drop(columns=['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2'], inplace=True, errors='ignore')
    df['event'] = df['event'].astype(str).str.strip().str.lower()
    df['event'] = df['event'].apply(lambda x: 0 if x == 'normal' else 1)
    
    return df

df= preprocess_data(df)

print("After preprocessing:")
print(df.head())
print(df['event'].unique())
print(df['event'].value_counts())  #Controlling that the convertions is performed correctly

#%% Normalize data, in two different ways to be able to use different feature selection methods

# Scaling using MinMaxScaler to adapt to Chi-square test that requires values >0
def scale_data_minmax(df):
    scaler = MinMaxScaler()
    features = df.drop(columns=['event'])
    df1 = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df1['event'] = df['event'].values
    return df1

df1= scale_data_minmax(df)

# Scaling using StandardScaler, for other methods e.g. LASSO
def standardize_data(df):
    scaler = StandardScaler()
    features = df.drop(columns=['event'])
    df_standardized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_standardized['event'] = df['event'].values
    return df_standardized

df = standardize_data(df)

print("After normalizing:")
print(df1.head())
print (df.head())

#%% Split data into train/test 80/20

X = df.drop(columns=['event'])
y = df['event']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X1 = df1.drop(columns=['event'])
y1 = df1['event']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

#%% Feature selection - filter methods

# Correlation analysis
correlation_with_event = df.corr()['event'].drop('event')
correlation_with_event_sorted = correlation_with_event.abs().sort_values(ascending=False)


print(correlation_with_event_sorted)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=correlation_with_event_sorted.index, 
    y=correlation_with_event_sorted.values,
    hue=correlation_with_event_sorted.index,
    palette="coolwarm")
plt.xticks(rotation=90)
plt.ylabel("Absolute correlation with event")
plt.xlabel("Features")
plt.title("Feature importance based on correlation")
plt.show()

# Chi-sqaure
def discretize_features(df1, num_bins=10):
    df_discretized = df1.copy()
    for col in df1.columns:
        if col != "event":  
            df_discretized[col] = pd.qcut(df1[col], q=num_bins, labels=False, duplicates="drop")
    return df_discretized

df_discretized = discretize_features(df1)

X_discretized = df_discretized.drop(columns=['event'])
y2 = df_discretized['event']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_discretized, y2, test_size=0.2, random_state=42)

def select_features_chi2(X_train1, y_train1):
    chi2_selector = SelectKBest(score_func=chi2, k=10)
    chi2_selector.fit(X_train1, y_train1)
    return chi2_selector.get_support(indices=True)

selected_features_chi2 = select_features_chi2(X_train2, y_train2)
print("Chi2 selected features:", X1.columns[selected_features_chi2])

# Select K-best, ANNOVA F-value
def select_features_f_classif(X_train, y_train):
    anova_selector = SelectKBest(score_func=f_classif, k=10)
    anova_selector.fit(X_train, y_train)
    return anova_selector.get_support(indices=True)

selected_features_f_classif = select_features_f_classif(X_train, y_train)
print("ANOVA selected features:", X.columns[selected_features_f_classif])

#%% Feature selection - wrapper method

# Recursive Feature selection
model = RandomForestClassifier(random_state=42)
selector = RFECV(estimator=model, step=1, cv=StratifiedKFold(10), scoring='accuracy')

selector.fit(X_train, y_train)

print("Selected features: ", X_train.columns[selector.support_])
print(f"Optimal number of features: {selector.n_features_}")

plt.figure()
plt.title("Recursive Feature Elimination")
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (accuracy)")
plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
plt.show()

# Create new train and test datasets
X_train_selected = X_train.iloc[:, selector.support_]
X_test_selected = X_test.iloc[:, selector.support_]

# Train and evaluate with the selected features
model.fit(X_train_selected, y_train)
print(f"Test accuracy: {model.score(X_test_selected, y_test)}")

#%% Feature selection- Embedded method

# LASSO
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)

# Get the coefficients and identify selected features
lasso_selected_features = np.where(lasso.coef_ != 0)[0]
print("Selected features by LASSO:", X.columns[lasso_selected_features])

# Random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]
print("Features ranked by importance:")
for i in range(X.shape[1]):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]}")