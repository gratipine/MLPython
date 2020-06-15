# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Sample classification with logisitc regression in Python
# 
# The data for this can be found https://www.kaggle.com/uciml/pima-indians-diabetes-database.

# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
diabetes_data = pd.read_csv("../../dataset/diabetes.csv")
diabetes_data.describe()

# %% [markdown]
# ### Preprocess

# %%
# You can do this if the Outcome is a string instead of a numeric
from sklearn import preprocessing

label_encoding = preprocessing.LabelEncoder()

diabetes_data["Outcome"] = label_encoding.fit_transform(diabetes_data["Outcome"].astype(str))


# %%
diabetes_data.sample(10)

# %% [markdown]
# ### Some plots

# %%
plt.figure(figsize=(8,8))
plt.scatter(diabetes_data["Glucose"], diabetes_data["Outcome"], c="g")
plt.xlabel("Glucose")
plt.ylabel("Outcome")

plt.show()


# %%
plt.figure(figsize=(8,8))
plt.scatter(diabetes_data["Age"], diabetes_data["Insulin"], c="g")
plt.xlabel("Age")
plt.ylabel("Insulin")

plt.show()


# %%
diabetes_data_correlation = diabetes_data.corr()
diabetes_data_correlation


# %%
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(diabetes_data_correlation, annot=True)

plt.show()

# %% [markdown]
# #### Rescaling

# %%
features = diabetes_data.drop("Outcome", axis=1)

from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler()


# %%
features_scaled = standard_scaler.fit_transform(features)

features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
features_scaled_df.head()


# %%
diabetes_data_scaled = pd.concat([features_scaled_df, diabetes_data["Outcome"]], axis=1)                                .reset_index(drop=True)

diabetes_data_scaled.to_csv("../dataset/diabetes_processed.csv")

# %% [markdown]
# ### Model fit and prediction

# %%
from sklearn.model_selection import train_test_split

X = diabetes_data_scaled.drop("Outcome", axis=1)
Y = diabetes_data_scaled["Outcome"]


# %%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# %%
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty='l2', C=1.0, solver='liblinear' )

classifier.fit(x_train, y_train)


# %%
y_pred = classifier.predict(x_test)


# %%
pred_results = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
pred_results

# %% [markdown]
# ### Evaluation
# 
# Using accuracy, recall and precision

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


