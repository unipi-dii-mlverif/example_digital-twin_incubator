import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import tqdm
import copy
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import confusion_matrix


 
torch.manual_seed(9999)
# Define the model
model = nn.Sequential(
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10,15),
    nn.ReLU(),
    nn.Linear(15,20),
    nn.ReLU(),
    nn.Linear(20,15),
    nn.ReLU(),
    nn.Linear(15,10),
    nn.ReLU(),
    nn.Linear(10,2),
    nn.Sigmoid()
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = pd.read_csv("../datasets/20230501_calibration_empty_system/20230501_calibration_empty_system.csv")

DESIRED_UL_TEMPERATURE = 38
DESIRED_LL_TEMPERATURE = DESIRED_UL_TEMPERATURE-2


Xdata = (dataset[["average_temperature"]]).values
Ydata = (dataset[["heater_on"]]).values

model.to(device) 
print(model)
 
print("Model built")
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005) 
print("Reading data");
 
print("Data read")

print(len(Xdata))

ul_tensor = (torch.ones([len(Xdata),1])*DESIRED_UL_TEMPERATURE).to(device)
ll_tensor = (torch.ones([len(Xdata),1])*DESIRED_LL_TEMPERATURE).to(device)

 
X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, train_size=0.80, shuffle=True, random_state=9999)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)

X_train = torch.hstack((X_train, ul_tensor[0:len(X_train)], ll_tensor[0:len(X_train)]))
y_train = torch.FloatTensor(y_train).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_test = torch.hstack((X_test, ul_tensor[0:len(X_test)], ll_tensor[0:len(X_test)]))
y_test = torch.FloatTensor(y_test).to(device)




test_number = len(y_test)
print(str(len(X_test)) + " " + str(len(y_test)) + " " + str(len(X_train)) + " " + str(len(y_train)))
 
# training parameters
n_epochs = 1000   # number of epochs to run
batch_size = int(len(X_train)/10)  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

print(f'Batch size {batch_size}, epochs {n_epochs}, train {len(X_train)}, test {len(X_test)}') 
 
# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
print("Starting training")
# training loop
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")
        running_loss = 0.0
        running_acc = 0.0
        running_count = 0
        for start in bar:
            # take a batch
            
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            optimizer.zero_grad()

            y_pred = model(X_batch)
            
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            loss.backward()
            running_loss+=loss.item()
            running_count+=1
            # update weights
            optimizer.step()

            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch

    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test).detach()
    print(y_pred[10:20])
    
    print("Epoch %i / %i test mse: %f" % (epoch+1, n_epochs, mse))
 
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
 
# restore model and return best accuracy
model.load_state_dict(best_weights)
torch.save(model, "incubator.pth")
