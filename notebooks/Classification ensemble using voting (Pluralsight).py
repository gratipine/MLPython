# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Overview 
# 
# A demo of how to set up a classifier ensemble in python using hard voting (majority) and soft voting (probability).
# %% [markdown]
# ### Data in

# %%
import pandas as pd

diabetes_data = pd.read_csv("../dataset/diabetes_processed.csv", index_col=0)


# %%
diabetes_data

# %% [markdown]
# ### Fit

# %%
X = diabetes_data.drop("Outcome", axis=1)
Y = diabetes_data["Outcome"]


# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# %%
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# %%
log_clf = LogisticRegression(C=1, solver="liblinear")

svc_clf = SVC(C=1, kernel="linear", gamma="auto")

naive_cls = GaussianNB()

# %% [markdown]
# #### Hard voting

# %%
voting_clf_hard = VotingClassifier(estimators = [("linear", log_clf),
                                                 ("SVC", svc_clf),
                                                 ("naive", naive_cls)],
                                  voting="hard")


# %%
voting_clf_hard.fit(x_train, y_train)


# %%
y_pred = voting_clf_hard.predict(x_test)


# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# %%
for clf_hard in (log_clf, svc_clf, naive_cls, voting_clf_hard):
    clf_hard.fit(x_train, y_train)
    
    y_pred = clf_hard.predict(x_test)
    print(clf_hard.__class__.__name__, accuracy_score(y_test, y_pred))

# %% [markdown]
# #### Soft

# %%
svc_soft = SVC(C=1, kernel="linear", gamma="auto", probability=True)

voting_clf_soft = VotingClassifier(estimators = [("linear", log_clf),
                                                 ("SVC", svc_soft),
                                                 ("naive", naive_cls)],
                                  voting="soft", 
                                  weights=[0.25, 0.5, 0.25])


# %%
for clf_soft in (log_clf, svc_soft, naive_cls, voting_clf_soft):
    clf_soft.fit(x_train, y_train)
    
    y_pred = clf_soft.predict(x_test)
    print(clf_soft.__class__.__name__, accuracy_score(y_test, y_pred))


