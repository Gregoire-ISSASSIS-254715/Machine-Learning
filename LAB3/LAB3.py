#################
##### LAB 3 #####
#################

#################
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy import quantile, where, random
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
#################

###############################################
##### Exercise 1 : SVM for Classification #####
###############################################

print('\nExercise 1 : SVM for Classification :\n')

#################
# Load Iris dataset
iris = load_iris()

# Get Iris feature names
print('\nIris feature names :')
print(iris.feature_names)

# Get Iris data between 0 & 5
print('\nIris data :')
print(iris.data[0:5, :])

# Get Iris target
print('\nIris target :')
print(iris.target[:])

# Get the whole Iris data
print('\nIris data (whole data) :')
print(iris.data)
#################

#################
# Split data into training and testing parts
print('\nSplit data into training and testing parts :')
X = iris.data
y = iris.target

# Specifies percentage of the data (here 20% of the data will be used for the testing and 80% for the checking)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Display X, X_train & X_test dimensions in console
print('X shape :', X.shape)
print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)
#################

#################
# Use a Support Vector Machine for classification
print('\nUse a Support Vector Machine for classification :')
SVMmodel = SVC(kernel='linear')  # Linear kernel will search for separate lines --> Definition of the kernel parameters
SVMmodel.fit(X_train, y_train)  # Feed the SVModel

# Check what are the parameters
SVMmodel.get_params()
print('SVM parameters :', SVMmodel.get_params())

# Check how good it classifies
SVMmodel.score(X_test, y_test)
print('SVM score :', SVMmodel.score(X_test, y_test))
#################

#################
# Plot scatters of targets 0 and 1 and check the separability of the classes
print('\nChoose only first two features (columns) of Iris data :')
# X = iris.data[:, 0:2]
# y = iris.target
X = iris.data[iris.target != 2, 0:2]
y = iris.target[iris.target != 2]

# Specifies percentage of the data (here 20% of the data will be used for the testing and 80% for the checking)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Display X, X_train & X_test dimensions in console
print('X shape (2 targets) :', X.shape)
print('X_train shape (2 targets) :', X_train.shape)
print('X_test shape (2 targets) :', X_test.shape)
print('y shape (2 targets) :', y.shape)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='green')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue')
# plt.scatter(X[y == 2, 0], X[y == 2, 1], color='cyan')

# Name figure axis
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")

# Display the figure
plt.show()

# Train and test the SVM classifier, play with regularization parameter C
print('\nTrain and test the SVM classifier, play with regularization parameter C :')
SVMmodel = SVC(kernel='linear', C=200)  # Definition of the kernel parameters
SVMmodel.fit(X_train, y_train)  # Feed the SVModel

# Check what are the parameters
SVMmodel.get_params()
print('SVM parameters (C = 200) :', SVMmodel.get_params())

# Check how good it classifies
SVMmodel.score(X_test, y_test)
print('SVM score (C = 200) :', SVMmodel.score(X_test, y_test))

# Initialize the support vectors
supvectors = SVMmodel.support_vectors_
print('\nSupport vectors :\n', supvectors)
print('\nSupport vectors shape :', supvectors.shape)
print('Support vectors X_train shape :', X_train.shape)
print('Support vectors y_train shape :', y_train.shape)

# Plot the support vectors here
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='green')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='blue')
# plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], color='cyan')
plt.scatter(supvectors[:, 0], supvectors[:, 1], color='red', marker='+', s=50)

# Separating line coefficients:
W = SVMmodel.coef_
b = SVMmodel.intercept_
xgr = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)

# Display values in console
print('\nLine coefficients :')
print('W[:, 0] :', W[:, 0])
print('W[:, 1] :', W[:, 1])
print('b :', b)

# Plot the figure
ygr = -W[:, 0] / W[:, 1] * xgr - b / W[:, 1]
plt.scatter(xgr, ygr)

# Name figure axis
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")

# Display the figure
plt.show()
#################


##################################################
##### Exercise 2 : Anomaly detection via SVM #####
##################################################

print('\nExercise 2 : Anomaly detection via SVM :\n')

#################
# Import one-class SVM and generate data (Gaussian blobs in 2D-plane) :
random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

# Plot the figure
plt.scatter(x[:,0], x[:,1])
plt.show()
#################

#################
# Train one-class SVM and plot the outliers (outputs of prediction being equal to -1) :
SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)

# Fit the matrix
SVMmodelOne.fit(x)

# Predict the model
pred = SVMmodelOne.predict(x)
anom_index = where(pred == -1)
values = x[anom_index]

# Plot one dot for each observation
plt.scatter(x[:, 0], x[:, 1])
plt.scatter(values[:, 0], values[:, 1], color='red')

# Rename axis
plt.axis('equal')

# Plot the figure
plt.show()
#################

#################
# Plot the support vectors

#################

#################
# What if we want to have a control what is outlier?
# Use e.g. 5% "quantile" to mark the outliers.
# Every point with lower score than threshold will be an outlier.

# Compute the log-likelihood of each sample under the model
scores = SVMmodelOne.score_samples(x)

# Set the threshold
thresh = quantile(scores, 0.01)
print(thresh)
index = where(scores <= thresh)
values = x[index]

# Plot one dot for each observation
plt.scatter(x[:, 0], x[:, 1])
plt.scatter(values[:, 0], values[:, 1], color='red')

# Rename axis
plt.axis('equal')

# Plot the figure
plt.show()
#################
