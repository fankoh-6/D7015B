# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:25:26 2025

@author: fankoh-6
"""


#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_regression, SelectKBest, RFECV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.metrics import mean_squared_error , mean_absolute_error, classification_report, confusion_matrix


#%% Importing data and understanding the properties of the dataset

data = pd.read_excel('train delays-1.xlsx')
print(data.shape)
print(data.isnull().sum())
print(data.dtypes)
print(data.nunique())

#%% Removing NaN-values and the empty column Unnamed, and the column only containing one value for teh whole dataset

data.dropna(subset=['Train number'], inplace=True)
data= data.drop(columns=["Reason code - level 1", "Unnamed: 0"])

print(data.isnull().sum())

#%% Checking for correlation between train mission adn Tågnr/Train ID

corr1= data["Train mission"].corr(data["Tågnr"])

print("Correlation between Train mission and Tågnr", corr1)

if corr1 >= 0.9:
    print('Train mission and Tågnr are strongly correlated, remove Tågnr')


duplicates = data.groupby("Train ID")["Train mission"].nunique()
duplicates = duplicates[duplicates > 1]  

if duplicates.empty:
    print('Train ID and Train mission are strongly correlated, no duplicate values are present')
    
#%% Removing column Tågnr

data= data.drop(columns=["Tågnr"])

print(data.shape) #Control that the column was removed, number of columns should be 13

#%% Controlling delay time distribution

min_delay= min(data["registered delay"])
max_delay= max(data["registered delay"])
print(min_delay, max_delay)

plt.hist(data["registered delay"], bins=15)
plt.yscale("log")
sns.catplot(x= data["registered delay"], kind="box")
plt.show()

#%% Creating a grouped column for delay data

bins= [0, 4, 6, 8, 10, 12, 15, 20, 25, 50, 350]
labels= ['3-4 min','5-6 min', '7-8 min', '9-10 min', '11-12 min', '13-15 min','16-20 min','21-25 min','26-50 min', '51-350 min']

data['delay_grouped']=pd.cut(data['registered delay'], bins= bins, labels= labels, right=True)

count_group= data['delay_grouped'].value_counts().sort_index()

print(count_group)

#%% Remove outliers in delay data above 50 min (51-350)

filtered_data = data[data['registered delay'] <= 50].copy()

#Controlling that the values over 50 was removed
print(max(filtered_data["registered delay"]))

#%% Checking if Route and Route number matches perfectly
Route_count = filtered_data['Route'].value_counts()
Routenr_count = filtered_data['Route number'].value_counts()

matching_values = (Route_count.reset_index(drop=True) == Routenr_count.reset_index(drop=True)).all()
print("Does every value have an exact match?", matching_values)

# Counting how many missing values are present in Route number
count_dash = (filtered_data['Route number'] == -1).sum()
print("Number of missing routes:", count_dash)

#%% Feature engineering, creating new columns

# Based on date, creating columns for Quarter, Month and Weekday
filtered_data["Date"] = pd.to_datetime(filtered_data["Date"])

filtered_data["Month"] = filtered_data["Date"].dt.month  
filtered_data["Weekday"] = filtered_data["Date"].dt.weekday 
filtered_data["Quarter"]=filtered_data["Date"].dt.quarter 

# Presenting number of delays for each category, including operator
quarter_count= filtered_data.groupby(filtered_data["Quarter"]).size()
month_count= filtered_data.groupby(filtered_data["Month"]).size()
weekday_count= filtered_data.groupby(filtered_data["Weekday"]).size()
operator_count=filtered_data.groupby(filtered_data["Operator"]).size()

print(quarter_count, month_count, weekday_count, operator_count)

      
#%% Data visualisation, registered delay count against different features

#Delay count per operator
operators = operator_count.index

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))  # Beräknar antal
        return f"{pct:.1f}%\n({count})"  # Visar både procent och antal
    return my_autopct


plt.pie(operator_count, labels=operators, autopct=make_autopct(operator_count), colors=["mediumpurple", "hotpink"])
plt.title("Delay count per operator")
plt.show()

#%% Delay count per Month and Quarter

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(month_count.index, month_count.values, color='skyblue')
axes[0].set_title('Delays per month')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Count')
axes[0].set_xticks(range(1, 13))
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

axes[1].bar(quarter_count.index, quarter_count.values, color='salmon')
axes[1].set_title('Delays per quarter')
axes[1].set_xlabel('Quarter')
axes[1].set_ylabel('Count')
axes[1].set_xticks(range(1, 5))
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#%% Delay count per Train mission, top 30/bottom 30

top_missions = data['Train mission'].value_counts().head(30) 
bottom_missions= data['Train mission'].value_counts().tail(30)

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
sns.barplot(x=top_missions.index, y=top_missions.values, color='lightgreen')
plt.xlabel("Train mission")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.title("Top 30 Train Missions")

plt.subplot(1, 2, 2)
sns.barplot(x=bottom_missions.index, y=bottom_missions.values, color='salmon')
plt.xlabel("Train mission")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.title("Bottom 30 Train Missions")

plt.tight_layout()
plt.show()

#%% Delay count per Train number, top 30/bottom 30

top_tnr = filtered_data['Train number'].value_counts().head(30) 
bottom_tnr= filtered_data['Train number'].value_counts().tail(30)

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
sns.barplot(x=top_tnr.index, y=top_tnr.values, color='lightskyblue')
plt.xlabel("Train number")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.title("Top 30 Train Numbers")

plt.subplot(1, 2, 2)
sns.barplot(x=bottom_tnr.index, y=bottom_tnr.values, color='plum')
plt.xlabel("Train number")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.title("Bottom 30 Train Numbers")

plt.tight_layout()
plt.show()

#%% Delay count per Place, top 30/bottom 30

top_place = filtered_data['Place'].value_counts().head(30) 
bottom_place= filtered_data['Place'].value_counts().tail(30)

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
sns.barplot(x=top_place.index, y=top_place.values, color='lightpink')
plt.xlabel("Place")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.title("Top 30 Places")

plt.subplot(1, 2, 2)
sns.barplot(x=bottom_place.index, y=bottom_place.values, color='gold')
plt.xlabel("Place")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.title("Bottom 30 Places")

plt.tight_layout()
plt.show()

#%% Encoding of non numerical columns
print("Columns in dataset:", filtered_data.columns)

# Categorical columns for Label Encoding (many unique values)
label_cols = ["Place", "Route", "Reason code", "Reason code Level 3", "Train number"]

# Categorical columns for One-Hot Encoding (few unique values)
onehot_cols = ["Operator", "Reason code Level 2"]

# Converting to datatype category for better performancee
filtered_data[label_cols] = filtered_data[label_cols].astype("category")
  
# Label Encoding 
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    filtered_data[col] = le.fit_transform(filtered_data[col])
    label_encoders[col] = le  

# One-Hot Encoding 
filtered_data = pd.get_dummies(filtered_data, columns=onehot_cols, drop_first=True)  
new_onehot_cols = [col for col in filtered_data.columns if any(base_col in col for base_col in onehot_cols)]

# Show the first rows with the new columns
print(filtered_data.head())
filtered_data.to_excel("Filtered data.xlsx", sheet_name="Sheet1")

#%% Division into train and test, applying filter methods to select proper features

features = ["Train mission", "Reason code Level 3","Reason code", "Route", "Place", "Train number", "Month", "Quarter", "Weekday"]+ new_onehot_cols
X = filtered_data[features]
y = filtered_data["registered delay"]  

# Divide data into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_data = filtered_data.select_dtypes(include=["number"])  
correlation_matrix = numeric_data.corr()
print(correlation_matrix["registered delay"].sort_values(ascending=False))

# ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=5)  # Select the top 5 features
X_new = selector.fit_transform(X_train, y_train)
feature_scores = pd.DataFrame({"Feature": X.columns, "Score": selector.scores_})
print(feature_scores.sort_values(by="Score", ascending=False))

# Mutual Information
mi = mutual_info_regression(X_train, y_train)
mi_scores = pd.DataFrame({"Feature": X.columns, "MI Score": mi})
print(mi_scores.sort_values(by="MI Score", ascending=False))

#%% Applying wrapper method RFE

new_onehot_cols1 = [col for col in filtered_data.columns if any(base_col in col for base_col in onehot_cols) and "Operator" not in col]
features1 = ["Train mission", "Reason code Level 3","Reason code", "Route", "Place", "Train number", "Month", "Quarter"]+ new_onehot_cols1
X = filtered_data[features1]
y = filtered_data["registered delay"]  

# Divide data into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42) #n_estimators 50, max depth for fater computation

# RFE with cross validation (CV)
selector = RFECV(rf, step=5, cv=3) #step 5 and CV3 for faster computation
selector.fit(X_train, y_train)

# Print the selected features
selected_features = X.columns[selector.support_]
print("Selected features:", selected_features)

#%% Applying embedded methods, RF importance and Lasso regression

features2 = ["Train mission", "Reason code Level 3","Reason code", "Route", "Place", "Train number", "Month"]
X = filtered_data[features2]
y = filtered_data["registered delay"]  

# Divide data into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest feature importance
rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Fetch and sort feature importance
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

print("Feature Importance (Embedded - RF):\n", feature_importance)

# Standardize data for Lasso Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso Regression 
lasso = LassoCV(cv=5, random_state=42)  
lasso.fit(X_train_scaled, y_train)

# Set features with coefficient ≠ 0 as important
lasso_selected_features = X.columns[lasso.coef_ != 0]
print("Important features according to Lasso:", lasso_selected_features)

#%% Linear regression, one feature

X = filtered_data[['Place']]
y = filtered_data["registered delay"]  

# Stratified train/test-split (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predction and evaluation of model
y_pred = lin_reg.predict(X_test)

print("Mean Absolute Error (Regression):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (Regression):", mean_squared_error(y_test, y_pred))
print("R-squared (Regression):", lin_reg.score(X_test, y_test))

#%% Multiple linear regression, best result

X = filtered_data[['Place', 'Reason code', 'Train mission','Reason code Level 3','Train number', 'Month', 'Route']]
y = filtered_data["registered delay"]  

# Train/test-split (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predction and evaluation of model
y_pred = lin_reg.predict(X_test)

print("Mean Absolute Error (Regression):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (Regression):", mean_squared_error(y_test, y_pred))
print("R-squared (Regression):", lin_reg.score(X_test, y_test))


#%% Random Forest regression

rf_reg = RandomForestRegressor(n_estimators=300, random_state=42)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

print("Mean Absolute Error (RF Regression):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (RF Regression):", mean_squared_error(y_test, y_pred))
print("R-squared (RF Regression):", rf_reg.score(X_test, y_test))


#%% Classification method for prediction, Logistic regression

features_cat = ["Train mission", "Reason code Level 3", "Reason code", "Route", "Place", "Train number", "Month"]
X = filtered_data[features_cat]
y = filtered_data["delay_grouped"]  

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)  

# Create and train regression model
model = LogisticRegression(max_iter=10000, class_weight= 'balanced')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
conf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#%% Classification method for prediction, Random Forrest

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()