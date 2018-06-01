import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection

# loading dataset directly from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']  # features
dataset = pandas.read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]  # data
y = array[:, 4]  # labels
validation_size = 0.20
seed = 7

# Splitting dataset into training and test set (X) with corresponding labels (y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size,
                                                                    random_state=seed)
