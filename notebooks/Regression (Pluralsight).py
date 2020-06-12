# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Regression
# 
# This notebook is a part of a course given by Janani Ravi at Pluralsight. The course name is "Designing a Machine Learning Model". 
# 
# The data can be found here: https://www.kaggle.com/sazid28/advertising.csv

# %%
import sklearn
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# %%
print(sklearn.__version__)

# %% [markdown]
# #### Data

# %%
advertising_data = pd.read_csv("../dataset/Advertising.csv", index_col=0)
advertising_data.head()


# %%
advertising_data.describe()

# %% [markdown]
# ### Analysis

# %%
plt.figure(figsize=(8,8))
plt.scatter(advertising_data["newspaper"], advertising_data["sales"], c="y")
plt.show()


# %%
plt.figure(figsize=(8,8))
plt.scatter(advertising_data["radio"], advertising_data["sales"], c="y")
plt.show()


# %%
plt.figure(figsize=(8,8))
plt.scatter(advertising_data["TV"], advertising_data["sales"], c="y")
plt.show()


# %%
advertising_data.corr()


# %%
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(advertising_data.corr(), annot=True)

# %% [markdown]
# ### Regression

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# %%
X = advertising_data["TV"].values.reshape(-1, 1)
Y = advertising_data["sales"].values.reshape(-1, 1)

X.shape, Y.shape


# %%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# %%
import statsmodels.api as sm

x_train_with_intercept = sm.add_constant(x_train)

stats_model = sm.OLS(y_train, x_train_with_intercept)

fit_model = stats_model.fit()

print(fit_model.summary())


# %%
linear_reg = LinearRegression(normalize=True).fit(x_train, y_train)


# %%
print("training:", linear_reg.score(x_train, y_train))


# %%
y_pred = linear_reg.predict(x_test)


# %%
from sklearn.metrics import r2_score
print("R squared score", r2_score(y_test, y_pred))


# %%
def adjusted_r2_score(r_square, labels, features):
    adj_r_square = 1 - ((1 - r_square) * (len(labels) - 1)) / (len(labels) - features.shape[1])
    
    return adj_r_square


# %%
print("Adjusted r2 score", adjusted_r2_score(r2_score(y_test, y_pred), y_test, x_test))


# %%
plt.figure(figsize=(8,8))
plt.scatter(x_test, y_test, c="black")
plt.plot(x_test, y_pred, c="blue", linewidth=2)
plt.xlabel("Money spent on TV")
plt.ylabel("Sales")
plt.show()

# %% [markdown]
# ### Multile regression

# %%
X = advertising_data.drop("sales", axis=1)
Y = advertising_data["sales"]


# %%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# %%
x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)

fit_model = stats_model.fit()
print(fit_model.summary())


# %%
linear_reg = LinearRegression(normalize=True).fit(x_train, y_train)
linear_reg


# %%
print("Training Score:", linear_reg.score(x_train, y_train))

# %% [markdown]
# #### With prediction

# %%
y_pred = linear_reg.predict(x_test)
print("Testing Score:", r2_score(y_test, y_pred))


