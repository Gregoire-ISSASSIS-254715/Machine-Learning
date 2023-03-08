#################
##### LAB 5 #####
#################

# THE LAB HAS BEEN DONE USING GOOGLE COLLAB
# THIS FILE HAS BEEN MADE USING PYCHARM WITH THE GOOGLE COLLAB COMMANDS FOR BETTER COMMENTS OF THE CODE

#################
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
#################

#################################################################
#################### Exercise 1 : XOR problem ###################
#################################################################

print('\nExercise 1 : XOR problem :\n')

# 1 : Prepare data
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# 2 : Creating the model
model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# 3 : Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 4 : Model training
history = model.fit(X, y, epochs=2000, batch_size=1, verbose=0)

# 5 : Model evaluation
loss, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy*100))

# 6 : Model predictions
for id_x, data_sample in enumerate(X):
  prediction = model.predict([data_sample])
  print(f"Data sample is {data_sample}, prediction from model {prediction}, ground_truth {y[id_x]}")

# 7 : Display loss function during the training process and accuracy
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('n epochs')
plt.ylabel('loss')

# TASK: Change these parameters and see the differences
# number of epochs
# learning_rate
# activation functions in layers
# batch_size
# verbose
# number of neurons in the hidden layer
#################


#################################################################
############# Exercise 2 : Congressional Voting Data ############
#################################################################

print('\nExercise 2 : Congressional Voting Data :\n')

# 1 : Join Google Collab and Google Drive
# from google.colab import drive        # Import the Google Drive into Google Collab
# drive.mount('/content/drive')         # Join the tools

# 2.1 : Loading dataset
path_to_dataset = '/content/drive/MyDrive/ML - LAB5/voting_complete.csv'    # Change the PATH
pd_dataset = pd.read_csv(path_to_dataset)                                   # Open the dataset

# 2.2 : Display the dataset
pd_dataset

# 3.1 : Define a function for train and test split
def train_test_split(pd_data: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    pd_dataset = pd_data.copy()                         # Work on a copy of the dataset
    pd_dataset = pd_dataset[pd_dataset.columns[1:]]     # Remove the first column (column index)
    index = np.arange(len(pd_dataset))                  # Give an array on a given interval
    index = np.random.permutation(index)                # Randomly permute a sequence, or return a permuted range
    train_ammount = int(len(index)*test_ratio)          # Initialize percentages of dataset to be trained and tested
    train_ids = index[train_ammount:]                   # Train on 80% of the dataset
    test_ids = index[:train_ammount]                    # Test on 20% of the dataset

    train_dataset = pd_dataset[pd_dataset.index.isin(train_ids)].reset_index()
    test_dataset = pd_dataset[pd_dataset.index.isin(test_ids)].reset_index()

    train_dataset = train_dataset[train_dataset.columns[1:]]
    test_dataset = test_dataset[test_dataset.columns[1:]]

    return train_dataset[train_dataset.columns[1:]], train_dataset[train_dataset.columns[0]], test_dataset[test_dataset.columns[1:]], test_dataset[test_dataset.columns[0]]

x_train, y_train, x_test, y_test = train_test_split(pd_dataset)

# 4 : Data examination
x_train

# Answers:
# 1. The performed task is a classification task.
#
# 2. As we can see when we display the pd_dataset in part 1, there are 435 rows. So, we have 435 samples in the dataset.
#
# 3. As for the previous question, we see that the pd_dataset contains 435 rows and 18 columns.
#    So we have 435*18 features in the dataset. Finally, there are 7830 features in the dataset.
#
# 4. We have 3 data types in our dataset:
#     - y : yes ;
#     - n : no ;
#     - ? : missing value (unknown).
#
# 5. Yes, there are missing values in the dataset. These values are represented by the "?" character.
#
# 6. In the dataset, we have 2 labels:
#     - 0 : n;
#     - 1 : y.

# 5 : Data preprocessing
X = pd.get_dummies(x_train)
print(X)

y_train = y_train.replace('republican', 1)
y = y_train.replace('democrat', 0)

# 6.1 : Creating the model
model = Sequential()
model.add(Dense(16, input_dim=48, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# 6.2 : Model summary
model.summary()

# 6.3 : Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 6.4 : Train the model
history = model.fit(X, y, epochs=2000, batch_size=500, verbose=0)

# 7.1 : First, apply the same preprocessing you did to train set to test set also

# 7.2 : Evaluate the model, print final accuracy and loss
loss, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy*100))
# Accuracy scores: 99.43, 99.71, 100, etc.

# 7.3 : Plot loss and validation loss depending on the training epochs into one graph.
# In another graph, plot accuracy and validation accuracy
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('n epochs')
plt.ylabel('loss')
