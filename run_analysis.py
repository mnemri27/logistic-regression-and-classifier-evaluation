
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from trivial_classifier import TrivialClassifier
from nbc_classifier import NBCclassifier
from logistic_classifier import LogisticClassifier


# Load datasets
data_folder = "Assignment3_Data"

datasets = {
    "A": (
        np.loadtxt(f"{data_folder}/Dataset3.2_A_X.csv", delimiter=","),
        np.loadtxt(f"{data_folder}/Dataset3.2_A_Y.csv", delimiter=",")
    ),
    "B": (
        np.loadtxt(f"{data_folder}/Dataset3.2_B_X.csv", delimiter=";"),
        np.loadtxt(f"{data_folder}/Dataset3.2_B_Y.csv", delimiter=";")
    )
}

training_sizes = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
repetitions = 100


for dataset_name, (X, Y) in datasets.items():

    print(f"\nDataset {dataset_name}")

    results = {
        "Trivial": [],
        "Naive Bayes": [],
        "Logistic Regression": []
    }

    for size in training_sizes:

        trivial_scores = []
        nbc_scores = []
        logistic_scores = []

        for i in range(repetitions):

            # Split dataset into 75% training and 25% testing
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.25, random_state=i
            )

            # Use only part of the training data
            X_train, _, Y_train, _ = train_test_split(
                X_train, Y_train,
                train_size=size,
                random_state=i
            ) if size < 1 else (X_train, None, Y_train, None)

            # Trivial Classifier
            trivial = TrivialClassifier()
            trivial.train(X_train, Y_train)

            predictions = trivial.predict(X_test)
            trivial_scores.append(accuracy_score(Y_test, predictions))

            # Naive Bayes
            if dataset_name == "A":
                categorical = list(range(X.shape[1]))
            else:
                categorical = []

            nbc = NBCclassifier(categorical_features=categorical)
            nbc.train(X_train, Y_train)

            predictions = nbc.predict(X_test)
            nbc_scores.append(accuracy_score(Y_test, predictions))

            # Logistic Regression
            if dataset_name == "A":
                encoder = OneHotEncoder(handle_unknown="ignore")
                X_train_lr = encoder.fit_transform(X_train)
                X_test_lr = encoder.transform(X_test)
            else:
                X_train_lr = X_train
                X_test_lr = X_test

            logistic = LogisticClassifier(L=1)
            logistic.train(X_train_lr, Y_train)

            predictions = logistic.predict(X_test_lr)
            logistic_scores.append(accuracy_score(Y_test, predictions))

        # Average accuracy over 100 repetitions
        results["Trivial"].append(np.mean(trivial_scores))
        results["Naive Bayes"].append(np.mean(nbc_scores))
        results["Logistic Regression"].append(np.mean(logistic_scores))

    # Plot results
    for name, accuracies in results.items():
        plt.plot(training_sizes, accuracies, marker="o", label=name)
        print(name, np.round(accuracies, 4))

    plt.title(f"Dataset {dataset_name}")
    plt.xlabel("Fraction of training data")
    plt.ylabel("Average test accuracy")
    plt.xticks(training_sizes)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"dataset_{dataset_name}_accuracy.png")
    plt.close()
