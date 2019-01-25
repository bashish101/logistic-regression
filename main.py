import numpy as np
from sklearn import datasets

from utils import *
from logistic_regressor import LogisticRegressor

def iris_classification():
	print('\nIris classification using Logistic Regression\n')
	print('Initiating Data Load...')

	iris = datasets.load_iris()
	# X, y = iris.data, iris.target
	# y = one_hot_encode(y)

	X, y = iris.data[iris.target != 2], iris.target[iris.target != 2]
	y = y.reshape(y.shape[0], 1)

	size = len(X)
	indices = list(range(size))
	np.random.shuffle(indices)
	X, y = np.array([X[idx] for idx in indices]), np.array([y[idx] for idx in indices])

	train_size = int(0.8 * len(X))
	X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

	print('Data load complete!')

	print('Constructing classifier...')
	size = (X_train.shape[-1], y_train.shape[-1])
	classifier = LogisticRegressor(size)
	classifier.fit(X_train, y_train)

	print('Generating test predictions...')
	predictions = classifier.predict(X)

	accuracy = np.sum([all(y_true == y_pred) for y_true, y_pred in zip(y, predictions)]) / len(predictions) * 100.
	print("Accuracy = {:.2f}%".format(accuracy))

def digit_recognition():
	print('\nDigit recognition using Logistic Regression\n')
	print('Initiating Data Load...')
	digits = datasets.load_digits()
	X, y = digits.data, digits.target

	pca = PCA()
	X = pca.transform(X, num_components = 23)
	y = one_hot_encode(y)

	size = len(X)
	indices = list(range(size))
	np.random.shuffle(indices)
	X, y = np.array([X[idx] for idx in indices]), np.array([y[idx] for idx in indices])

	train_size = int(0.8 * len(X))
	X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

	print('Constructing classifier...')
	size = (X_train.shape[-1], y_train.shape[-1])
	classifier = LogisticRegressor(size)
	classifier.fit(X_train, y_train)

	print('Generating test predictions...')
	predictions = classifier.predict(X)

	accuracy = np.sum([all(y_true == y_pred) for y_true, y_pred in zip(y, predictions)]) / len(predictions) * 100.
	print("Accuracy = {:.2f}%".format(accuracy))


if __name__ == '__main__':
	np.random.seed(3)

	iris_classification()
	digit_recognition()
