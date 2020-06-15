# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import seaborn as sns

from sklearn import datasets

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d


# %%
X, color = datasets.make_swiss_roll(n_samples=2000)


# %%
X = pd.DataFrame(X)

ax = plt.subplots(figsize = (8, 8))
ax = plt.axes(projection = "3d")

ax.scatter3D(X[0], X[1], X[2], c=color)#, cmap=plt.cm.Spectral)
plt.show()

# %% [markdown]
# ### Reduction

# %%
def apply_manifold_learning(X, method):
    X = method.fit_transform(X)
    
    X = pd.DataFrame(X)
    
    plt.subplots(figsize=(8,8))
    plt.axis("equal")
    
    plt.scatter(X[0], X[1], c=color)
    
    return method


# %%
from sklearn.manifold import MDS
mds = apply_manifold_learning(X, MDS(n_components=2, metric=False))


# %%
from sklearn.manifold import MDS
mds = apply_manifold_learning(X, MDS(n_components=2, metric=True))


# %%
from sklearn.manifold import LocallyLinearEmbedding

# how do you check these?
lle = apply_manifold_learning(
    X, LocallyLinearEmbedding(n_neighbors=15, n_components=2, method="standard"))


# %%
from sklearn.manifold import LocallyLinearEmbedding

# how do you check these?
lle = apply_manifold_learning(
    X, LocallyLinearEmbedding(n_neighbors=15, n_components=2, method="hessian"))


