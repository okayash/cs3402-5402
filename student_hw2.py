import math
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter, defaultdict
# Do not import any other libraries. You can use built-in functions and the above imports only.


###################### Task-1 ################################################

def entropy(labels):
    """
    Compute entropy of a list of class labels.
    """
    # TODO: implement entropy

    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    entropy = 0.0
    for i in counts.values():
        p = i / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy



def information_gain(dataset):
    """
    Input:
        dataset: list of lists
                 last column is label
    Output:
        list of information gain for each feature
    """
    ig_list = []
    # compute IG for each feature
    # TODO: implement information gain
    ig_list = []

    if not dataset:
        return ig_list

    n = len(dataset)
    num_features = len(dataset[0]) - 1

    labels = [row[-1] for row in dataset]
    H_parent = entropy(labels)

    for feature_idx in range(num_features):
        partitions = defaultdict(list)
        for row in dataset:
            partitions[row[feature_idx]].append(row[-1])
        H_children = 0.0
        for subset_labels in partitions.values():
            weight = len(subset_labels) / n
            H_children += weight * entropy(subset_labels)

        ig_list.append(H_parent - H_children)

    return ig_list



###################### Task-2 ################################################
def perceptron_gradient_descent(X, y, w_init, b_init, lr=1.0, max_iter=100):
    """
    Parameters:
        X : list of feature vectors
        y : list of labels (-1 or +1)
        w_init : initial weight vector
        b_init : initial bias
        lr : learning rate
        max_iter : maximum iterations
        
    Returns:
        w, b
    """
    # TODO: implement perceptron learning algorithm with gradient descent

    w = list(w_init)
    b = b_init
    n = len(X)

    for _ in range(max_iter):
        misclassified = []
        for i in range(n):
            dot = sum(w[j] * X[i][j] for j in range(len(w)))
            if y[i] * (dot + b) <= 0:
                misclassified.append(i)
        if not misclassified:
            break
        chosen = max(misclassified)
        for j in range(len(w)):
            w[j] += lr * y[chosen] * X[chosen][j]
        b += lr * y[chosen]
        
    return w, b

###################### Task-3 ################################################
from sklearn.svm import SVC
import random

np.random.seed(42)
random.seed(42)

X_pos = np.random.randn(25, 2) * 0.6 + np.array([2.0, 2.0])
X_neg = np.random.randn(25, 2) * 0.6 + np.array([-2.0, -2.0])

X_clean = np.vstack([X_pos, X_neg])
y_clean = np.array([1] * 25 + [-1] * 25)

X_list = X_clean.tolist()
y_list = y_clean.tolist()
random.seed(0)

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', label='Class +1', zorder=3)
ax.scatter(X_neg[:, 0], X_neg[:, 1], color='green', label='Class -1', zorder=3)

x_range = np.linspace(-5, 5, 300)

for _ in range(10):
    w_init = [random.uniform(-1, 1), random.uniform(-1, 1)]
    b_init = random.uniform(-1, 1)
    w, b = perceptron_gradient_descent(X_list, y_list, w_init, b_init, lr=1.0, max_iter=1000)
    y_line = -(w[0] * x_range + b) / w[1]
    ax.plot(x_range, y_line, color='black', linewidth=1, alpha=0.6)
clf = SVC(kernel='linear', C=1000)
clf.fit(X_clean, y_clean)
w_svm = clf.coef_[0]
b_svm = clf.intercept_[0]

y_svm = -(w_svm[0] * x_range + b_svm) / w_svm[1]
ax.plot(x_range, y_svm, color='red', linewidth=2.5, label='SVM (C=1000)')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Parts 2 & 3: Perceptron (black) vs SVM (red)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
plt.tight_layout()
plt.savefig('part2_3_comparison.png', dpi=150)
plt.show()
y_noisy = y_clean.copy()
np.random.seed(7)
flip_pos = np.random.choice(np.where(y_clean == 1)[0], size=5, replace=False)
flip_neg = np.random.choice(np.where(y_clean == -1)[0], size=5, replace=False)
y_noisy[flip_pos] = -1
y_noisy[flip_neg] = 1
for C in [0.01, 0.1, 1, 10, 100]:
    clf_c = SVC(kernel='linear', C=C)
    clf_c.fit(X_clean, y_noisy)
    w_c = clf_c.coef_[0]
    b_c = clf_c.intercept_[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_clean[y_noisy == 1, 0], X_clean[y_noisy == 1, 1], color='blue', label='Class +1 (noisy)', zorder=3)
    ax.scatter(X_clean[y_noisy == -1, 0], X_clean[y_noisy == -1, 1], color='green', label='Class -1 (noisy)', zorder=3)

    y_line = -(w_c[0] * x_range + b_c) / w_c[1]
    ax.plot(x_range, y_line, color='red', linewidth=2, label=f'SVM C={C}')
    ax.plot(x_range, -(w_c[0] * x_range + b_c - 1) / w_c[1], color='orange', linewidth=1, linestyle='--', label='Margins')
    ax.plot(x_range, -(w_c[0] * x_range + b_c + 1) / w_c[1], color='orange', linewidth=1, linestyle='--')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f'Part 6: Soft-Margin SVM  C={C}  (noisy labels)')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'part6_C{C}.png', dpi=150)
    plt.show()