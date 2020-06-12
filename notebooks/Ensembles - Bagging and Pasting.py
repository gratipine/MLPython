# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd

# %% [markdown]
# ### Data in 

# %%
insurance_data = pd.read_csv("./../dataset/insurance_preprocessed.csv")


# %%
X = insurance_data.drop("charges", axis=1)
Y = insurance_data["charges"]


# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# %% [markdown]
# ### ML

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


# %%
bag_reg = BaggingRegressor(
    DecisionTreeRegressor(),
    n_estimators=500,
    bootstrap=True, #makes it bagging rather than pasting
    max_samples=0.8,
    n_jobs=-1, #max possible in the system
    oob_score=True) # out of bag score

bag_reg.fit(x_train, y_train)
                           


# %%
bag_reg.oob_score_


# %%
from sklearn.metrics import r2_score
y_pred = bag_reg.predict(x_test)
r2_score(y_test, y_pred)


# %%
bag_reg = BaggingRegressor(
    DecisionTreeRegressor(),
    n_estimators=500,
    bootstrap=False, #makes it bagging rather than pasting
    max_samples=0.9,
    n_jobs=-1 #max possible in the system
) 

bag_reg.fit(x_train, y_train)
                           


# %%
from sklearn.metrics import r2_score
y_pred = bag_reg.predict(x_test)
r2_score(y_test, y_pred)


