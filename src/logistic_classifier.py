
from sklearn.linear_model import LogisticRegression


class LogisticClassifier:
    def __init__(self, L=1):
        if L <= 0:
            raise ValueError("L must be positive.")

        self.L = L
        self.clf = LogisticRegression(
            C=1 / L,
            max_iter=2000
        )

    def train(self, X, Y):
        self.clf.fit(X, Y)
        return self

    def predict(self, X):
        return self.clf.predict(X)
