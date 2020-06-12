# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# %%
diabetes_data = pd.read_csv("./../dataset/diabetes_processed.csv", index_col=0)


# %%
diabetes_data


# %%
FEATURES = list(diabetes_data.columns[:-1])
FEATURES

# %% [markdown]
# #### PCA

# %%
from sklearn.decomposition import PCA

def apply_pca(n):
    pca = PCA(n_components=n)
    x_new = pca.fit_transform(diabetes_data[FEATURES])
    
    return pca, pd.DataFrame(x_new)


# %%
pca_obj, _ = apply_pca(8)


# %%
print("Explained variance: ", pca_obj.explained_variance_ratio_)


# %%
plt.figure(figsize=(8, 8))

plt.plot(np.cumsum(pca_obj.explained_variance_ratio_))

plt.xlabel("n components")
plt.ylabel("Explained variance")
plt.show()


# %%
Y = diabetes_data["Outcome"]
_, X_new = apply_pca(5)

# %% [markdown]
# #### Train on the reduced X set

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_new, Y, test_size = 0.3)


# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear').fit(x_train, y_train)


# %%
y_pred = model.predict(x_test)

# %% [markdown]
# #### Evaluation

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


