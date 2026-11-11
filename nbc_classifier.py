
import numpy as np

from mixed_naive_bayes import MixedNB
from sklearn.naive_bayes import GaussianNB


class NBCclassifier:
    def __init__(self, categorical_features=None, alpha=1):
        self.categorical_features = categorical_features
        self.alpha = alpha
        self.clf = None

    def train(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y, dtype=int)

        if self.categorical_features is None:
            categorical = list(range(X.shape[1]))
        else:
            categorical = list(self.categorical_features)

        if len(categorical) == 0:
            self.clf = GaussianNB()
        else:
            self.clf = MixedNB(
                categorical_features=categorical,
                max_categories=[10] * len(categorical),
                alpha=self.alpha
            )

        self.clf.fit(X, Y)
        return self

    def predict(self, X):
        return self.clf.predict(np.asarray(X))
