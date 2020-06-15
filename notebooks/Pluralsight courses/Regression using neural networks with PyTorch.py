# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
advertising_data = pd.read_csv("./../dataset/Advertising.csv", index_col=0)


# %%
advertising_data


# %%
from sklearn import preprocessing

# scaling as neural networks work better with small data
advertising_data["TV"] = preprocessing.scale(advertising_data["TV"])
advertising_data["radio"] = preprocessing.scale(advertising_data["radio"])
advertising_data["newspaper"] = preprocessing.scale(advertising_data["newspaper"])


# %%
X = advertising_data.drop("sales", axis=1)
Y = advertising_data["sales"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


# %%
import torch

x_train_tensor = torch.tensor(x_train.values, dtype=torch.float)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)


# %%
inp = 3 # dimension of input layer
out = 1 # dimension of output layer
hid = 100 # hidden layers
loss_fun = torch.nn.MSELoss()

learning_rate = 0.0001 # how much the model learns during each epoch


# %%
model = torch.nn.Sequential(torch.nn.Linear(inp, hid),
                           torch.nn.ReLU(),
                           torch.nn.Linear(hid, out))


# %%
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# %%
number_epochs = 10000

for iter in range(number_epochs):
    y_pred = model(x_train_tensor)
    
    loss = loss_fun(y_pred, y_train_tensor)
    
    if iter % 1000 == 0:
        print(iter, loss.item()) # show what the loss is at that point
        
    optimizer.zero_grad() # zero out the gradients, since they get accumulated
    # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
    loss.backward() # calculate the gradients to apply to the model's parameters
    
    optimizer.step() # apply the gradients to the model


# %%
y_pred_tensor = model(x_test_tensor)
y_pred_tensor[:5]


# %%
y_pred = y_pred_tensor.detach().numpy()


# %%
plt.figure(figsize=(8, 8))
plt.scatter(y_pred, y_test)

plt.xlabel("predicted")
plt.ylabel("actual")


# %%
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


