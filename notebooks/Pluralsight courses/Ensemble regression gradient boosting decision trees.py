# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Data in

# %%
import pandas as pd


# %%
insurance_data = pd.read_csv("./../dataset/insurance_preprocessed.csv")


# %%
X = insurance_data.drop("charges", axis=1)
Y = insurance_data["charges"]


# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# %% [markdown]
# ### Gradient boost the long way

# %%
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=3)
tree_reg1.fit(x_train, y_train)


# %%
y2 = y_train - tree_reg1.predict(x_train)


# %%
tree_reg2 = DecisionTreeRegressor(max_depth=3)
tree_reg2.fit(x_train, y2)
y3 = y2 - tree_reg2.predict(x_train)


# %%
tree_reg3 = DecisionTreeRegressor(max_depth=3)
tree_reg3.fit(x_train, y3)


# %%
y_pred = sum(tree.predict(x_test) for tree in (tree_reg1, tree_reg2, tree_reg3))


# %%
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# %% [markdown]
# ### Gradient boost the short way

# %%
from sklearn.ensemble import GradientBoostingRegressor


# %%
gbr = GradientBoostingRegressor(max_depth=3, n_estimators=3, learning_rate=1.0)
gbr.fit(x_train, y_train)


# %%
y_pred = gbr.predict(x_test)
r2_score(y_test, y_pred)


