import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold


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

# Fill TEST missing values (train already handled above)
test["Age"]  = test["Age"].fillna(train["Age"].median())      # or test["Age"].median()
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Embarked"] = test["Embarked"].fillna(train["Embarked"].mode()[0])



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

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

#defining the parameter grid
param_grid = {
    "n_estimators": [200, 400, 600], #number of trees
    "max_depth": [None, 5, 7, 9], #max level per tree
    "min_samples_split": [2, 5, 10], #min amount of samples to split a node
    "min_samples_leaf": [1, 2, 4], #min samples in a leaf
    "max_features": ["sqrt", "log2", 0.5], #number of features tried at each split
    "class_weight": [None, "balanced"], #handle class imbalance
    "bootstrap": [True], #sample rows with replacement for each tree
}

#do cross validation with k=5 folds and shuffle being true, its stratified so the proportion of labels will stay the same in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1, #shows some output while running grid search
)

grid.fit(X, y)
print("Best CV accuracy: ", grid.best_score_)
print("Best params: ", grid.best_params_)



"""rf = RandomForestClassifier(
    n_estimators=300, #number of trees
    max_depth=None, #max level per tree
    min_samples_split=2, #min amount of samples to split a node
    min_samples_leaf=1, #min samples in a leaf
    max_features="sqrt", #number of features tried at each split
    random_state=42, 
    n_jobs=-1 #parallelism, set -1 to use all cores
)"""

#initialize rf with the best parameters found
best_rf = grid.best_estimator_
#fit the best rf on the data

test_for_model = test[feature_columns]
test_pred = best_rf.predict(test_for_model)

raw_test = pd.read_csv('./Datasets/test.csv')
submission = pd.DataFrame({
    "PassengerId": raw_test["PassengerId"],
    "Survived": test_pred
})
submission.to_csv("submission_randomForest_withHPO.csv", index=False)



