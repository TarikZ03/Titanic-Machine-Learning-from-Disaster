import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


train = pd.read_csv('./Datasets/train.csv')
test = pd.read_csv('./Datasets/test.csv')

#Filling in missing values

#fill in Age with the median age because mean is to sensitive to outliers
train["Age"].fillna(train["Age"].median(), inplace=True)

#since Cabin is missing in almost all fields we replace it with a binary feature indicating 1=has cabin and 0=doesnt have cabin and a categorical feature indicating the deck of the cabin
for df in [train, test]:
    df["HasCabin"] = df["Cabin"].notna().astype(int) #1 if cabin info exists, 0 if it doenst
    df["CabinDeck"] = df["Cabin"].str[0].fillna("U") #first letter of Cabin is the deck, U=unknown

#fill in the missing values with mode
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)

#drop the raw cabin after creating the HasCabin/CabinDeck combination
for df in [train, test]:
    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)


#convert categorical columns into numerical features by one-hot encoding
categorical_columns = ["Sex", "Embarked", "CabinDeck"]
train = pd.get_dummies(train, columns=categorical_columns, drop_first=True)
test = pd.get_dummies(test, columns=categorical_columns, drop_first=True)

#make the test dataset have the same columns as the train set after one hot encoding and etc
all_feature_columns = [c for c in train.columns if c != "Survived"]
test = test.reindex(columns=all_feature_columns, fill_value=0)

drop_from_features = {"PassengerId", "Name", "Ticket"}
feature_columns = [c for c in all_feature_columns if c not in drop_from_features]

#create a train/val split
X = train[feature_columns]
y = train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#scaling the numerical features so that each feature has mean=0 and sd=1
numerical_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
scaler = StandardScaler()
#make copies of the datasets because we want to be able to go back to the original if something happens
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
#fit the scaler, the scaler learns the mean and sd of each feature and then applies the transform
X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
#here we use the .transform only because we dont want to compute a new mean/sd from the validation set because we must apply the same scaling learned from training data to the validation data
X_val_scaled[numerical_features] = scaler.transform(X_val[numerical_features])

#make the logistic predictions
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

#prediction scores
y_pred = log_reg.predict(X_val_scaled)
print("Accuracy", accuracy_score(y_val, y_pred))
print("Classification report:", classification_report(y_val, y_pred))
print("Confusion matrix: ", confusion_matrix(y_val, y_pred))