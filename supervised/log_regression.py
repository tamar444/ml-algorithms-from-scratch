import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, epochs=800):
        self.lr=lr
        self.epochs=epochs
        self.weights=None
        self.bias=None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features,1))
        self.bias = 0
        y=y.reshape((n_samples,1))
        losses = []

        for _ in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            eps = 1e-9
            loss = -np.mean(y*np.log(y_pred+eps) + (1-y)*np.log(1-y_pred+eps))
            losses.append(loss)

            dw = (1/n_samples)*np.dot(X.T,(y_pred-y))
            db = (1/n_samples)*np.sum((y_pred-y))

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

        return self.weights, self.bias, losses

    def predict_proba(self,X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self,X,threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)