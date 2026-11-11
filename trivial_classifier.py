
class TrivialClassifier:
    def __init__(self):
        self.most_repeated_class = None

    def train(self, X, Y):
        count = {}

        for y in Y:
            count[y] = count.get(y, 0) + 1

        self.most_repeated_class = max(count, key=count.get)

    def predict(self, X):
        return [self.most_repeated_class] * len(X)
