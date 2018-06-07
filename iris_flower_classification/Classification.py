import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_models():

    """
    Trains the models on test sets using 10-fold cross validation
    Calculates a score for each model
    Prints mean and standard deviation for the score
    """

    print("Training results, mean, and standard deviation: \n")
    for name, model in models:
        # Creates 10 folds, each fold is used once for validation while the other 9 are used for training
        kfold = model_selection.KFold(n_splits=10)
        # Evaluates score by cross validation. model is the classifier, X_train is the data to fit,
        # y_train is the target data to predict, and cv=kfold is an iterable yielding train/test splits
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
        training_results.append(cv_results)
        names.append(name)
        print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))


def train_and_test_models():

    """
    Uses the test set to calculate the accuracy of the trained models. Score is saved in a list.
    Prints the accuracy, confusion matrix, and classification report.
    """

    for name, model in models:
        m = model
        # X_train is the training vectors, y_train is the target values/labels
        m.fit(X_train, y_train)
        # Performs prediction on the test vectors, and returns a list
        predictions = m.predict(X_test)
        # Computes the subset accuracy as a float
        # y_test is the true labels, predictions are the predicted values.
        score = accuracy_score(y_test, predictions)
        test_results.append((name, score))
        print("\n\nAccuracy of the {}: {}".format(name, score))
        print("\nConfusion matrix: ")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification report: ")
        print(classification_report(y_test, predictions))
        print()


# loading dataset directly from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']  # features
data_set = pandas.read_csv(url, names=names)

# Split-out validation dataset
array = data_set.values
X = array[:, 0:4]  # data/features
y = array[:, 4]  # labels
test_size = 0.20

# Splitting dataset into training and test set with corresponding labels
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

# Models
models = [('Logistic Regression', LogisticRegression()), ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
          ('K-Nearest Neighbors', KNeighborsClassifier()), ('Decision Tree Classifier', DecisionTreeClassifier()),
          ('Gaussian Naive Bayes', GaussianNB()), ('Support Vector Machine', SVC())]

training_results, names = [], []
test_results = []

evaluate_models()
train_and_test_models()

test_results.sort(key=lambda x: x[1], reverse=True)

print("Results from best to worst: ")
for n, v in test_results:
    print("Model: {}, Accuracy: {}".format(n, v))
