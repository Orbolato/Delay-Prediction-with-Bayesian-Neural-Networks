import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt
import random

# Importing data preprocessed in R
final_data_sbgr = pd.read_csv(r'C:\Users\lucas\Desktop\Doutorado\Dados\Paper Delay Prediction\final_data_sbgr.csv')

# Removing categorical data
final_data_sbgr = final_data_sbgr.drop('sg_empresa_icao', axis='columns')
final_data_sbgr = final_data_sbgr.drop('sg_icao_origem', axis='columns')

# Removing columns filled with zeros
for col in final_data_sbgr.columns:
    if all(final_data_sbgr.loc[:,col] == 0):
        final_data_sbgr = final_data_sbgr.drop(col, axis='columns')
        
print('Case 10')

# Plotting boxplot
bp = final_data_sbgr.boxplot(column='en_route_delay')
bp.plot()
plt.show()

# Splitting the data into input and output data
x = final_data_sbgr.iloc[:,0:(len(final_data_sbgr.columns)-1)]
y = final_data_sbgr.iloc[:,len(final_data_sbgr.columns)-1]

# Shuffling data
x, y = shuffle(x, y)

# Splitting the data into training and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, 
                                                    train_size=.75,
                                                    shuffle=True)

# Data scaling
scaler = MinMaxScaler()
data = pd.concat([x_train, y_train], axis=1)
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
x_train = scaled_data.iloc[:,0:(len(scaled_data.columns)-1)]
y_train = scaled_data.iloc[:,len(scaled_data.columns)-1]
test_data = pd.DataFrame(scaler.transform(pd.concat([x_test, y_test], axis=1)), 
                         columns=data.columns)
x_test = test_data.iloc[:,0:(len(test_data.columns)-1)]
y_test = test_data.iloc[:,len(test_data.columns)-1]

# Converting the data to run in the Torch BNN model
x_train = torch.Tensor(x_train.values)
y_train = torch.Tensor(y_train.values)
y_train = torch.unsqueeze(y_train, dim=1)
x_test = torch.Tensor(x_test.values)
y_test = torch.Tensor(y_test.values)
y_test = torch.unsqueeze(y_test, dim=1)

# Creating BNN model
model = nn.Sequential(
    nn.Linear(in_features=len(x.columns), 
              out_features=2*len(x.columns) + 1, 
              bias=True),
    nn.Linear(in_features=2*len(x.columns) + 1, 
              out_features=len(x.columns), 
              bias=True),
    nn.Linear(in_features=len(x.columns), 
              out_features=1, 
              bias=True),
)

# Defining error metrics
mse_loss = nn.MSELoss()

# Initializing the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the model
print('Training...')
mse_list = []
n = 10000
for step in range(n):
    
    # print('Training step ' + str(step + 1))
    
    # Predictions
    pre = model(x_train)
    mse = mse_loss(pre, y_train)
    mse_list.append(mse.item())
    
    # Model optimization
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

# Print the results of the trainig process
print('- Training MSE: %2.2E' % (mse.item()))

# Testing the model
print('Testing...')
pre = model(x_test)
mse_test = mse_loss(pre, y_test)

# Print the results of the test process
print('- Test MSE: %2.2E' % (mse_test.item()))

# Plotting learning process
plt.plot(mse_list)
plt.xlabel('Steps')
plt.ylabel('MSE')
plt.title('ANN training process')
plt.show()

# Getting the test results
print('Plotting test results...')
models_result = model(x_test).data.numpy().T[0]
models_result = scaler.data_min_[len(scaler.data_min_)-1] + models_result*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])
mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
std_values = np.array([models_result[i].std() for i in range(len(models_result))])
y_test_orig = y_test.data.numpy().T[0]
y_test_orig = scaler.data_min_[len(scaler.data_min_)-1] + y_test_orig*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])

# Plotting test outputs
plt.figure()
plt.plot(range(len(y_test_orig)),y_test_orig,'.',color='darkorange',markersize=2,label='Test set')
plt.plot(range(len(y_test_orig)),models_result,color='navy',lw=1,label='Predicted Mean Model')
plt.plot(range(len(y_test_orig)),np.repeat(np.mean(y_test_orig),len(y_test_orig)),'--',color='red',lw=1,label='Real Mean')
plt.legend()
plt.xlabel('index')
plt.ylabel('y')
plt.title('Test process')
plt.show()

# Print the results of the test process
print('- Test mean: %2.2f' % (np.mean(models_result)))
print('- Real test mean: %2.2f' % (np.mean(y_test_orig)))

# Getting the training results
print('Plotting training results...')
models_result_train = model(x_train).data.numpy().T[0]
models_result_train = scaler.data_min_[len(scaler.data_min_)-1] + models_result_train*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])
mean_values_train = np.array([models_result_train[i].mean() for i in range(len(models_result_train))])
std_values_train = np.array([models_result_train[i].std() for i in range(len(models_result_train))])
y_train_orig = y_train.data.numpy().T[0]
y_train_orig = scaler.data_min_[len(scaler.data_min_)-1] + y_train_orig*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])

# Plotting training outputs
plt.figure()
plt.plot(range(len(y_train_orig)),y_train_orig,'.',color='darkorange',markersize=2,label='Training set')
plt.plot(range(len(y_train_orig)),mean_values_train,color='navy',lw=1,label='Predicted Mean Model')
plt.plot(range(len(y_train_orig)),np.repeat(np.mean(y_train_orig),len(y_train_orig)),'--',color='red',lw=1,label='Real Mean')
plt.legend()
plt.xlabel('index')
plt.ylabel('y')
plt.title('Training process')
plt.show()

# Print the results of the training process
print('- Training mean: %2.2f' % (np.mean(mean_values_train)))
print('- Real training mean: %2.2f' % (np.mean(y_train_orig)))

###############################################################################

print('Case 11')

# Removing outliers
Q1 = final_data_sbgr['en_route_delay'].quantile(0.25)
Q3 = final_data_sbgr['en_route_delay'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 3*IQR
upper = Q3 + 3*IQR
remove = np.where((final_data_sbgr['en_route_delay'] > upper) | 
                  (final_data_sbgr['en_route_delay'] < lower))[0]
final_data_sbgr.drop(index=remove, inplace=True, axis='rows')
final_data_sbgr.reset_index(drop=True, inplace=True)
plt.figure()
plt.hist(final_data_sbgr['en_route_delay'])
plt.show()

# Plotting boxplot
plt.figure()
bp = final_data_sbgr.boxplot(column='en_route_delay')
bp.plot()
plt.show()

# Splitting the data into input and output data
x = final_data_sbgr.iloc[:,0:(len(final_data_sbgr.columns)-1)]
y = final_data_sbgr.iloc[:,len(final_data_sbgr.columns)-1]

# Shuffling data
x, y = shuffle(x, y)

# Splitting the data into training and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, 
                                                    train_size=.75,
                                                    shuffle=True)

# Data scaling
scaler = MinMaxScaler()
data = pd.concat([x_train, y_train], axis=1)
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
x_train = scaled_data.iloc[:,0:(len(scaled_data.columns)-1)]
y_train = scaled_data.iloc[:,len(scaled_data.columns)-1]
test_data = pd.DataFrame(scaler.transform(pd.concat([x_test, y_test], axis=1)), 
                         columns=data.columns)
x_test = test_data.iloc[:,0:(len(test_data.columns)-1)]
y_test = test_data.iloc[:,len(test_data.columns)-1]

# Converting the data to run in the Torch BNN model
x_train = torch.Tensor(x_train.values)
y_train = torch.Tensor(y_train.values)
y_train = torch.unsqueeze(y_train, dim=1)
x_test = torch.Tensor(x_test.values)
y_test = torch.Tensor(y_test.values)
y_test = torch.unsqueeze(y_test, dim=1)

# Creating BNN model
model = nn.Sequential(
    nn.Linear(in_features=len(x.columns), 
              out_features=2*len(x.columns) + 1, 
              bias=True),
    nn.Linear(in_features=2*len(x.columns) + 1, 
              out_features=len(x.columns), 
              bias=True),
    nn.Linear(in_features=len(x.columns), 
              out_features=1, 
              bias=True),
)

# Defining error metrics
mse_loss = nn.MSELoss()

# Initializing the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the model
print('Training...')
mse_list = []
n = 10000
for step in range(n):
    
    # print('Training step ' + str(step + 1))
    
    # Predictions
    pre = model(x_train)
    mse = mse_loss(pre, y_train)
    mse_list.append(mse.item())
    
    # Model optimization
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

# Print the results of the trainig process
print('- Training MSE: %2.2E' % (mse.item()))

# Testing the model
print('Testing...')
pre = model(x_test)
mse_test = mse_loss(pre, y_test)

# Print the results of the test process
print('- Test MSE: %2.2E' % (mse_test.item()))

# Plotting learning process
plt.plot(mse_list)
plt.xlabel('Steps')
plt.ylabel('MSE')
plt.title('ANN training process')
plt.show()

# Getting the test results
print('Plotting test results...')
models_result = model(x_test).data.numpy().T[0]
models_result = scaler.data_min_[len(scaler.data_min_)-1] + models_result*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])
mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
std_values = np.array([models_result[i].std() for i in range(len(models_result))])
y_test_orig = y_test.data.numpy().T[0]
y_test_orig = scaler.data_min_[len(scaler.data_min_)-1] + y_test_orig*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])

# Plotting test outputs
plt.figure()
plt.plot(range(len(y_test_orig)),y_test_orig,'.',color='darkorange',markersize=2,label='Test set')
plt.plot(range(len(y_test_orig)),models_result,color='navy',lw=1,label='Predicted Mean Model')
plt.plot(range(len(y_test_orig)),np.repeat(np.mean(y_test_orig),len(y_test_orig)),'--',color='red',lw=1,label='Real Mean')
plt.legend()
plt.xlabel('index')
plt.ylabel('y')
plt.title('Test process')
plt.show()

# Print the results of the test process
print('- Test mean: %2.2f' % (np.mean(models_result)))
print('- Real test mean: %2.2f' % (np.mean(y_test_orig)))

# Getting the training results
print('Plotting training results...')
models_result_train = model(x_train).data.numpy().T[0]
models_result_train = scaler.data_min_[len(scaler.data_min_)-1] + models_result_train*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])
mean_values_train = np.array([models_result_train[i].mean() for i in range(len(models_result_train))])
std_values_train = np.array([models_result_train[i].std() for i in range(len(models_result_train))])
y_train_orig = y_train.data.numpy().T[0]
y_train_orig = scaler.data_min_[len(scaler.data_min_)-1] + y_train_orig*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])

# Plotting training outputs
plt.figure()
plt.plot(range(len(y_train_orig)),y_train_orig,'.',color='darkorange',markersize=2,label='Training set')
plt.plot(range(len(y_train_orig)),mean_values_train,color='navy',lw=1,label='Predicted Mean Model')
plt.plot(range(len(y_train_orig)),np.repeat(np.mean(y_train_orig),len(y_train_orig)),'--',color='red',lw=1,label='Real Mean')
plt.legend()
plt.xlabel('index')
plt.ylabel('y')
plt.title('Training process')
plt.show()

# Print the results of the training process
print('- Training mean: %2.2f' % (np.mean(mean_values_train)))
print('- Real training mean: %2.2f' % (np.mean(y_train_orig)))

###############################################################################

print('Case 12')

# Turning the histogram into an uniform distribution
hist = plt.hist(final_data_sbgr['en_route_delay'])
n = int(min(hist[0]))
for i in range(len(hist[0])):
    total = np.where((final_data_sbgr['en_route_delay'] >= hist[1][i]) & 
                     (final_data_sbgr['en_route_delay'] < hist[1][i+1]))[0]
    m = int(hist[0][i] - n)
    if m > 0:
        remove = random.sample(sorted(total), k=m)
        final_data_sbgr.drop(remove, inplace=True, axis='rows')
        final_data_sbgr.reset_index(drop=True, inplace=True)
plt.figure()
plt.hist(final_data_sbgr['en_route_delay'], edgecolor='black', color='grey')
plt.xlabel('En-route delay (min)')
plt.ylabel('Frequency')
plt.title('En-route delay histogram (SBGR)')
plt.show()

# Plotting boxplot
plt.figure()
bp = final_data_sbgr.boxplot(column='en_route_delay')
bp.plot()
plt.show()

# Splitting the data into input and output data
x = final_data_sbgr.iloc[:,0:(len(final_data_sbgr.columns)-1)]
y = final_data_sbgr.iloc[:,len(final_data_sbgr.columns)-1]

# Shuffling data
x, y = shuffle(x, y)

# Splitting the data into training and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, 
                                                    train_size=.75,
                                                    shuffle=True)

# Data scaling
scaler = MinMaxScaler()
data = pd.concat([x_train, y_train], axis=1)
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
x_train = scaled_data.iloc[:,0:(len(scaled_data.columns)-1)]
y_train = scaled_data.iloc[:,len(scaled_data.columns)-1]
test_data = pd.DataFrame(scaler.transform(pd.concat([x_test, y_test], axis=1)), 
                         columns=data.columns)
x_test = test_data.iloc[:,0:(len(test_data.columns)-1)]
y_test = test_data.iloc[:,len(test_data.columns)-1]

# Converting the data to run in the Torch BNN model
x_train = torch.Tensor(x_train.values)
y_train = torch.Tensor(y_train.values)
y_train = torch.unsqueeze(y_train, dim=1)
x_test = torch.Tensor(x_test.values)
y_test = torch.Tensor(y_test.values)
y_test = torch.unsqueeze(y_test, dim=1)

# Creating BNN model
model = nn.Sequential(
    nn.Linear(in_features=len(x.columns), 
              out_features=2*len(x.columns) + 1, 
              bias=True),
    nn.Linear(in_features=2*len(x.columns) + 1, 
              out_features=len(x.columns), 
              bias=True),
    nn.Linear(in_features=len(x.columns), 
              out_features=1, 
              bias=True),
)

# Defining error metrics
mse_loss = nn.MSELoss()

# Initializing the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the model
print('Training...')
mse_list = []
n = 10000
for step in range(n):
    
    # print('Training step ' + str(step + 1))
    
    # Predictions
    pre = model(x_train)
    mse = mse_loss(pre, y_train)
    mse_list.append(mse.item())
    
    # Model optimization
    optimizer.zero_grad()
    mse.backward()
    optimizer.step()

# Print the results of the trainig process
print('- Training MSE: %2.2E' % (mse.item()))

# Testing the model
print('Testing...')
pre = model(x_test)
mse_test = mse_loss(pre, y_test)

# Print the results of the test process
print('- Test MSE: %2.2E' % (mse_test.item()))

# Plotting learning process
plt.plot(mse_list)
plt.xlabel('Steps')
plt.ylabel('MSE')
plt.title('ANN training process')
plt.show()

# Getting the test results
print('Plotting test results...')
models_result = model(x_test).data.numpy().T[0]
models_result = scaler.data_min_[len(scaler.data_min_)-1] + models_result*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])
mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
std_values = np.array([models_result[i].std() for i in range(len(models_result))])
y_test_orig = y_test.data.numpy().T[0]
y_test_orig = scaler.data_min_[len(scaler.data_min_)-1] + y_test_orig*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])

# Plotting test outputs
plt.figure()
plt.plot(range(len(y_test_orig)),y_test_orig,'.',color='darkorange',markersize=2,label='Test set')
plt.plot(range(len(y_test_orig)),models_result,color='navy',lw=1,label='Predicted Mean Model')
plt.plot(range(len(y_test_orig)),np.repeat(np.mean(y_test_orig),len(y_test_orig)),'--',color='red',lw=1,label='Real Mean')
plt.legend()
plt.xlabel('index')
plt.ylabel('y')
plt.title('Test process')
plt.show()

# Print the results of the test process
print('- Test mean: %2.2f' % (np.mean(models_result)))
print('- Real test mean: %2.2f' % (np.mean(y_test_orig)))

# Getting the training results
print('Plotting training results...')
models_result_train = model(x_train).data.numpy().T[0]
models_result_train = scaler.data_min_[len(scaler.data_min_)-1] + models_result_train*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])
mean_values_train = np.array([models_result_train[i].mean() for i in range(len(models_result_train))])
std_values_train = np.array([models_result_train[i].std() for i in range(len(models_result_train))])
y_train_orig = y_train.data.numpy().T[0]
y_train_orig = scaler.data_min_[len(scaler.data_min_)-1] + y_train_orig*(scaler.data_max_[len(scaler.data_min_)-1] - scaler.data_min_[len(scaler.data_min_)-1])

# Plotting training outputs
plt.figure()
plt.plot(range(len(y_train_orig)),y_train_orig,'.',color='darkorange',markersize=2,label='Training set')
plt.plot(range(len(y_train_orig)),mean_values_train,color='navy',lw=1,label='Predicted Mean Model')
plt.plot(range(len(y_train_orig)),np.repeat(np.mean(y_train_orig),len(y_train_orig)),'--',color='red',lw=1,label='Real Mean')
plt.legend()
plt.xlabel('index')
plt.ylabel('y')
plt.title('Training process')
plt.show()

# Print the results of the training process
print('- Training mean: %2.2f' % (np.mean(mean_values_train)))
print('- Real training mean: %2.2f' % (np.mean(y_train_orig)))
