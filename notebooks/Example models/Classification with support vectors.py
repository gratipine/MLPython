# %%
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv("../data/diabetes.csv")

#%% 
features = data.drop("Outcome", axis=1)

from sklearn import preprocessing
standard_scaler = preprocessing.StandardScaler()

features_scaled = standard_scaler.fit_transform(features)

features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# %%
prediction_column = "Outcome"
Y = data[prediction_column]
X = features_scaled_df

# %%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# %%
clf = svm.SVC()
clf.fit(x_train, y_train)

# %%
y_pred = clf.predict(x_test)


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
