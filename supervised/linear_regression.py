import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=800):
        self.lr=lr
        self.epochs=epochs
        self.weights=None
        self.bias=None
        self.losses=[]

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features,1))
        self.bias = 0
        y=y.reshape((n_samples,1))


        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_pred-y)**2)
            self.losses.append(loss)

            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))*2
            db = (1/n_samples)*np.sum((y_pred-y))*2

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

        return self.weights, self.bias, self.losses

    def predict(self,X):
        return np.dot(X, self.weights) + self.bias





        