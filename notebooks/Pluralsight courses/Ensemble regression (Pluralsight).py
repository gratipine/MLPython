# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Overview
# 
# This notebook hhas a look at insurance data and does regression on it.
# The data was found at this address: https://www.kaggle.com/mirichoi0218/insurance
# %% [markdown]
# ### Data in

# %%
import pandas as pd
import matplotlib.pyplot as plt

insurance_data = pd.read_csv("../dataset/insurance.csv")

# %% [markdown]
# #### Correlations

# %%
import seaborn as sns
insurance_data_correlation = insurance_data.corr()

fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(insurance_data_correlation, annot=True)
plt.show()

# %% [markdown]
# ### Preprecessing

# %%
from sklearn import preprocessing

label_encoding = preprocessing.LabelEncoder()


# %%
insurance_data["region"] = label_encoding.fit_transform(insurance_data["region"].astype(str))


# %%
label_encoding.classes_


# %%
insurance_data = pd.get_dummies(insurance_data, columns=['sex', 'smoker'])


# %%
insurance_data.to_csv("../dataset/insurance_preprocessed.csv", index=False)


