import numpy as np

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        separated = [[x for x,t in zip(X,y) if t==c] for c in self.classes_]
        count_sample = X.shape[0]

        #p(class)
        self.class_log_prior = [np.log(len(i)/count_sample) for i in separated]

        # word counts + smoothing
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha

        # log p(word | class)
        self.feature_log_prob = np.log(count / count.sum(axis=1)[np.newaxis].T)

    def predict_log_proba(self, X):
        return np.array([
            (self.feature_log_prob * x).sum(axis=1) + self.class_log_prior
            for x in X
        ])

    def predict(self, X):
        idx = np.argmax(self.predict_log_proba(X), axis=1)
        return self.classes_[idx]