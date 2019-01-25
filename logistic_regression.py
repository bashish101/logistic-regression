import numpy as np

class LogisticRegression(object):
	def __init__(self,
		     size,
		     optimization = 'gradient_descent',
		     lr = 0.1):
		self.size = size
		self.optimization = optimization
		self.lr = lr

		self.weights = None
		self.bias = None
		self.initialize_weights()

	def initialize_weights(self):
		limit = 1 / (self.size[0] ** 0.5)
		self.weights = np.random.uniform(-limit, limit, (self.size[0], self.size[1]))
		self.bias = np.random.uniform(-limit, limit, (self.size[1],))

	def shuffle_data(self, X, y):
		size = len(X)
		indices = list(range(size))
		np.random.shuffle(indices)
		shuffled_X, shuffled_y = np.array([X[idx] for idx in indices]), np.array([y[idx] for idx in indices])
		return shuffled_X, shuffled_y

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def sigmoid_prime(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))
	
	def cost(self, h, y):
		return np.mean(np.sum((h - y) ** 2, axis = 1), axis = 0)

	def gradient(self, h, y, z):
		# Partial derivative of cost (mse) with respect to parameters (weights or biases)
		h_prime = self.sigmoid_prime(z)
		delta = (h - y) * h_prime
		return delta

	def display_progress(self, progress, msg = None):
		completed = '#' * progress
		remaining = ' ' * (100 - progress)
		
		print ('\r[{0}{1}] {2}% | {3}'.format(completed, remaining, progress, msg), end = '\r')

	def fit(self, X, y, epochs = 10000, max_batch_size = 32):
		n = len(X)

		for epoch in range(epochs):
			err = 0.
			X, y = self.shuffle_data(X, y)
			mini_batches =  [(X[index:index + max_batch_size], y[index:index + max_batch_size]) \
					 for index in range(0, n, max_batch_size)]

			batch_count = len(mini_batches)
			for batch_idx, (X_batch, y_batch) in enumerate(mini_batches):
				batch_size = X_batch.shape[0]

				z = np.dot(X_batch, self.weights) + self.bias
				h = self.sigmoid(z)
				if self.optimization == 'gradient_descent':
					delta = self.gradient(h, y_batch, z)
					nabla_w = np.dot(X_batch.T, delta)
					nabla_b = np.dot(np.ones((batch_size,)), delta)
					self.weights -= (self.lr / batch_size) * nabla_w
					self.bias -= (self.lr / batch_size) * nabla_b
				cost = self.cost(h, y_batch)
				err += cost / batch_size

			progress = int((epoch / epochs) * 100)
			msg = "Average Train Loss: {:.4f}".format(err / batch_count)
			self.display_progress(progress, msg)

		print('\nTraining complete!')

	def predict(self, X):
		y = np.round(self.sigmoid(np.dot(X, self.weights) + self.bias)).astype(int)
		return y
