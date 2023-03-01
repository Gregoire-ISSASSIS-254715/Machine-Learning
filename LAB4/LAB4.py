#################
##### LAB 4 #####
#################

#################
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#################

#################################################################
##### Exercise 1 : PCA through Singular Value Decomposition #####
#################################################################

print('\nExercise 1 : PCA through Singular Value Decomposition :\n')

# Define 3 points matrix in 2D-space and its transposed matrix:
X = np.array([[2, 1, 0], [4, 3, 0]])
Xt = np.transpose(X)

# Display the matrix
print('X =\n', X)
print('\nXt =\n', Xt)

# Calculate the covariance matrix:
R = (1/3)*np.matmul(X, Xt)
print('\nR =\n', R)

# Calculate the SVD decomposition and new basis vectors:
[U, D, V] = np.linalg.svd(R)  # Call SVD decomposition
u1 = U[:, 0]    # New basis vectors
u2 = U[:, 1]

# Calculate the coordinates in new orthonormal basis:
Xi1 = np.matmul(Xt, u1)
Xi2 = np.matmul(Xt, u2)

# Display the coordinates
print('\nXi1 =', Xi1)
print('Xi2 =', Xi2)

# Calculate the approximation of the original from new basis
print('\nSecond matrix dimension :\n', Xi1[:, None])     # Add second dimension to array and test it

XApprox = np.matmul(u1[:, None], Xi1[None, :])      # Approximation
print('\nOriginal approximation :\n', XApprox)

# Check that you got the original
XOrigin = np.matmul(u1[:, None], Xi1[None, :]) + np.matmul(u2[:, None], Xi2[None, :])
print('\nOriginal matrix :\n', XOrigin)
print()
#################


#########################################
##### Exercise 2 : PCA on Iris data #####
#########################################

print('\nExercise 2 : PCA on Iris data :\n')

# Load Iris dataset as in the last PC lab:
iris = load_iris()

# Get Iris feature names
print('Iris feature names :')
print(iris.feature_names)

# Get Iris data between 0 & 5
print('\nIris data :')
print(iris.data[0:5, :])

# Get Iris target
print('\nIris target :')
print(iris.target[:])

# We have 4 dimensions of data, plot the first three columns in 3D
X = iris.data
y = iris.target

# Initialize axes
axes1 = plt.axes(projection='3d')
axes1.scatter3D(X[y == 0, 1], X[y == 0, 1], X[y == 0, 2], color='green')
axes1.scatter3D(X[y == 1, 1], X[y == 1, 1], X[y == 1, 2], color='blue')
axes1.scatter3D(X[y == 2, 1], X[y == 2, 1], X[y == 2, 2], color='magenta')

# Display the graphic
plt.show()

# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)
Xscaler = StandardScaler()
Xpp = Xscaler.fit_transform(X)

# Define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print('\nPCA Covariance :\n', pca.get_covariance())

# Initialize axes
axes2 = plt.axes(projection='3d')
axes2.scatter3D(Xpca[y == 0, 0], Xpca[y == 0, 1], Xpca[y == 0, 2], color='green')
axes2.scatter3D(Xpca[y == 1, 0], Xpca[y == 1, 1], Xpca[y == 1, 2], color='blue')
axes2.scatter3D(Xpca[y == 2, 0], Xpca[y == 2, 1], Xpca[y == 2, 2], color='magenta')

# Display the graphic
plt.show()

# Compute pca.explained_variance_ value
explainedVariance = pca.explained_variance_
print('\nExplained variance =', explainedVariance)

# Compute pca.explained_variance_ratio value
explainedVarianceRatio = pca.explained_variance_ratio_
print('Explained variance ratio =', explainedVarianceRatio)

# Plot the principal components in 2D, mark different targets in color
plt.scatter(Xpca[y == 0, 0], Xpca[y == 0, 1], color='green')
plt.scatter(Xpca[y == 1, 0], Xpca[y == 1, 1], color='blue')
plt.scatter(Xpca[y == 2, 0], Xpca[y == 2, 1], color='magenta')

# Display the graphic
plt.show()
print()
#################


#######################################
##### Exercise 3 : KNN classifier #####
#######################################

print('\nExercise 3 : KNN classifier :\n')

# Part 1 : Specifies percentage of the data (here 30% of the data will be used for the testing and 70% for the checking)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print('X_train shape :', X_train.shape)
print('X_test shape :', X_test.shape)

# Choose the needed number of neighbors
knn1 = KNeighborsClassifier(n_neighbors=3)

# Fit the matrix
knn1.fit(X_train, y_train)

# Compute the Ypred
Ypred = knn1.predict(X_test)

# Show confusion matrix
confusion_matrix(y_test, Ypred)
ConfusionMatrixDisplay.from_predictions(y_test, Ypred)
plt.show()
#################

#################
# Part 2 : Now do the same (data set split, KNN, confusion matrix),
# but for PCA-transformed data (1st two principal components, i.e., first two columns).
# Compare the results with full dataset
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X, y, test_size=0.3)
print('\nX_train_PCA shape :', X_train_pca.shape)
print('X_test_PCA shape :', X_test_pca.shape)

# Choose the needed number of neighbors
knn1 = KNeighborsClassifier(n_neighbors=3)

# Fit the matrix
knn1.fit(X_train_pca, y_train_pca)

# Compute the Ypred
Ypred_pca = knn1.predict(X_test_pca)

# Show confusion matrix
confusion_matrix(y_test_pca, Ypred_pca)
ConfusionMatrixDisplay.from_predictions(y_test_pca, Ypred_pca)
plt.show()
#################

#################
# Part 3 : Now do the same, but use only 2-dimensional data of original X (first two columns)
X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(X[:, 0:1], y, test_size=0.3)
print('\nX_train_PCA shape :', X_train_pca.shape)
print('X_test_PCA shape :', X_test_pca.shape)

# Choose the needed number of neighbors
knn1 = KNeighborsClassifier(n_neighbors=3)

# Fit the matrix
knn1.fit(X_train_wrong, y_train_wrong)

# Compute the Ypred
Ypred_wrong = knn1.predict(X_test_wrong)

# Show confusion matrix
confusion_matrix(y_test_wrong, Ypred_wrong)
ConfusionMatrixDisplay.from_predictions(y_test_wrong, Ypred_wrong)
plt.show()
#################
