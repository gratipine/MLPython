# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Sample classification with Decision trees in Python
# 
# The data for this can be found https://www.kaggle.com/uciml/pima-indians-diabetes-database.

# %%
import pandas as pd
import matplotlib.pyplot as plt

diabetes_data = pd.read_csv("../dataset/diabetes.csv")
diabetes_data.describe()

# %% [markdown]
# ### Rescaling

# %%
features = diabetes_data.drop("Outcome", axis=1)

from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler()

features_scaled = standard_scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

diabetes_data_scaled = pd.concat([features_scaled_df, diabetes_data["Outcome"]], axis=1)                                .reset_index(drop=True)

diabetes_data_scaled.to_csv("../dataset/diabetes_processed.csv")

# %% [markdown]
# ### Sample splitting

# %%
from sklearn.model_selection import train_test_split

X = diabetes_data_scaled.drop("Outcome", axis=1)
Y = diabetes_data_scaled["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# %% [markdown]
# ### Model fit

# %%
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=4)
classifier.fit(x_train, y_train)

# %% [markdown]
# ### Prediction and evaluation

# %%
y_pred = classifier.predict(x_test)


# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score

model_accuracy = accuracy_score(y_test, y_pred)
model_precision = precision_score(y_test, y_pred)
model_recall = recall_score(y_test, y_pred)

print("How many of the predicted labels were correct?")
print("Accuracy: ", model_accuracy)
print("")

print("How many of the positive predictions were correct?")
print("Precision: ", model_precision)
print("")

print("How many of the individuals with diabetes in the dataset were correclty classified?")
print("Recall: ", model_recall)


# %%
diabetes_crosstab = pd.crosstab(y_pred, y_test)
diabetes_crosstab

# %% [markdown]
# #### Manual calculation

# %%
TP = diabetes_crosstab[1][1]
FP = diabetes_crosstab[0][1]
TN = diabetes_crosstab[0][0]
FN = diabetes_crosstab[1][0]


# %%
accuracy_score = (TP + TN) / (TP + TN + FP + FN)
accuracy_score


# %%
precision_score = TP / (TP + FP)
precision_score


# %%
recall_score = TP / (TP + FN)
recall_score


