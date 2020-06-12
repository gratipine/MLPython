# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Perform clustering using K-means
# 
# The data specified in the course was unavailable (https://raw.githubusercontent.com/datascienceinc/learn-data-science/master/Introduction-to-K-means-Clustering/Data/data_1024.csv). I am using this one instead https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
# 
# Try to cluster spenders together.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_in = pd.read_csv("../dataset/Mall_Customers.csv")


# %%
data_in.head()


# %%
# shuffle so that order does not affect anything
data_in = data_in.sample(frac=1)


# %%
data_in.drop("CustomerID", axis=1, inplace=True)

# %% [markdown]
# ### Initial plots
# 
# Gender does not seem to be all that helpful.

# %%
data_in.plot(x="Annual Income (k$)", y="Spending Score (1-100)", kind="scatter")


# %%
fig, ax = plt.subplots()
plt.scatter(data_in.loc[data_in["Gender"] == "Female", "Annual Income (k$)"], 
            data_in.loc[data_in["Gender"] == "Female", "Spending Score (1-100)"], color="r")
plt.scatter(data_in.loc[data_in["Gender"] != "Female", "Annual Income (k$)"], 
            data_in.loc[data_in["Gender"] != "Female", "Spending Score (1-100)"], color="b")
plt.show()


# %%
data_in.plot(x="Age", y="Spending Score (1-100)", kind="scatter")


# %%
fig, ax = plt.subplots()
plt.scatter(data_in.loc[data_in["Gender"] == "Female", "Age"], 
            data_in.loc[data_in["Gender"] == "Female", "Spending Score (1-100)"], color="r")
plt.scatter(data_in.loc[data_in["Gender"] != "Female", "Age"], 
            data_in.loc[data_in["Gender"] != "Female", "Spending Score (1-100)"], color="b")
plt.show()


# %%
data_in.drop("Gender", axis=1, inplace=True)

# %% [markdown]
# ### Clustering

# %%
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=6, max_iter=1000).fit(data_in)


# %%
kmeans_model.labels_

# %% [markdown]
# #### Grab the labels

# %%
zipped_list = list(zip(np.array(data_in), kmeans_model.labels_))


# %%
centroids = kmeans_model.cluster_centers_


# %%
centroids

# %% [markdown]
# #### Plot the clusters
# 
# Note - this takes a second to render and is only for two dimensions - need to find a way to do a 3D graph

# %%
colors = ['g', 'y', 'b', 'k', 'r']
plt.figure(figsize=(10,8))

for element in zipped_list:
    plt.scatter(element[0][0], element[0][1], c=colors[(element[1] % len(colors))])
    
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=200, marker="s")

for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0], centroids[i][1]), fontsize=20)

# %% [markdown]
# Nice clear separation of the clusters, but I hope that the centroids are just bad plotting on my part.

# %%
colors = ['g', 'y', 'b', 'k', 'r', 'o']
plt.figure(figsize=(10,8))

for element in zipped_list:
    plt.scatter(element[0][1], element[0][2], c=colors[(element[1] % len(colors))])
    
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=200, marker="s")

for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][1], centroids[i][2]), fontsize=20)

# %% [markdown]
# ### Analysis
# 
# Score for 5 clusters is 0.44428597560893024.
# Score for 6 clusters is, but the clusters are not clearly separated.

# %%
from sklearn.metrics import silhouette_score
print ("Silhoutte score", silhouette_score(data_in, kmeans_model.labels_))


