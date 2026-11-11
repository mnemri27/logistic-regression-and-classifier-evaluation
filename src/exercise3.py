
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


X = np.loadtxt("Assignment3_Data/Dataset3.3_X.csv", delimiter=",")
Y = np.loadtxt("Assignment3_Data/Dataset3.3_Y.csv", delimiter=",").astype(int)

lambdas = [0, 0.5, 10, 100]
weights = []

for L in lambdas:
    C = np.inf if L == 0 else 1 / L

    model = LogisticRegression(
        C=C,
        max_iter=10000
    )

    model.fit(X, Y)

    w = np.concatenate((
        model.intercept_,
        model.coef_.ravel()
    ))

    weights.append(w)

    print(f"\nLambda = {L}")
    print("Weights:", np.round(w, 4))


# Use the same y-axis range for all plots
all_weights = np.concatenate(weights)
y_min = all_weights.min()
y_max = all_weights.max()
margin = max((y_max - y_min) * 0.1, 0.1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, L, w in zip(axes.flat, lambdas, weights):
    ax.plot(range(len(w)), w, marker="o")
    ax.axhline(0, color="gray", linestyle="--")

    ax.set_title(f"Lambda = {L}")
    ax.set_xlabel("Weight index")
    ax.set_ylabel("Weight value")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.grid(True)

plt.tight_layout()
plt.savefig("results/exercise3_weights.png")
plt.close()

print("\nPlot saved as exercise3_weights.png")
