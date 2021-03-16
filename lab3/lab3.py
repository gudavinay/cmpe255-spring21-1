import numpy as np
import pandas as pd
import os

np.random.seed(42)

# To plot pretty figures
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGE_DIR = "."


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(IMAGE_DIR, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def random_digit(X):
    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    save_fig("some_digit_plot")
    plt.show()
    return some_digit


def load_and_sort():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
        mnist = sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    return mnist["data"], mnist["target"]


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data = mnist.data.to_numpy()
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
    return mnist


def train_predict(some_digit, X_train, X_test, y_train, y_test):
    import numpy as np
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    # Example: Binary number 5 Classifier
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import mean_squared_error as rmse
    # DONE
    # print prediction result of the given input some_digit
    sgd = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
    sgd.fit(X_train, y_train_5)
    print('Predicting result of random number generated is 5: ', sgd.predict([some_digit]))
    y_pred_5 = sgd.predict(X_test)
    print('RMSE value: ', rmse(pd.DataFrame(y_test_5), pd.DataFrame(y_pred_5)))
    print('Cross validation score: ', calculate_cross_val_score(sgd, X_train, y_train_5))


def calculate_cross_val_score(sgd_clf, X_train, y_train_5):
    # DONE
    from sklearn.model_selection import cross_val_score
    return cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')


if __name__ == "__main__":
    X, y = load_and_sort()
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    some_digit = random_digit(X)
    train_predict(some_digit, X_train, X_test, y_train, y_test)
