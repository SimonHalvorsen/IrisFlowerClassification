import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def build_and_evaluate_models():

    """
    Trains the models on test sets using 10-fold cross validation
    Calculates a score for each model
    Prints mean and standard deviation for the score
    """

    print("Training results, mean, and standard deviation: ")
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)  # 10-fold cross validation
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
        training_results.append(cv_results)
        names.append(name)
        print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))


def test_models():

    """
    Uses the test set to calculate the accuracy of the trained models. Score is saved in a list.
    Prints the accuracy, confusion matrix, and classification report.
    """

    for name, model in models:
        m = model
        m.fit(X_train, y_train)
        predictions = m.predict(X_test)
        score = accuracy_score(y_test, predictions)
        test_results.append((name, score))
        print("\nAccuracy of the {}: {}".format(name, score))
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
X = array[:, 0:4]  # data
y = array[:, 4]  # labels
validation_size = 0.20
seed = 7

# Splitting dataset into training and test set (X) with corresponding labels (y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size,
                                                                    random_state=seed)
# Algorithms
models = [('Logistic Regression', LogisticRegression()), ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
          ('K-Nearest Neighbors', KNeighborsClassifier()), ('Decision Tree Classifier', DecisionTreeClassifier()),
          ('Gaussian Naive Bayes', GaussianNB()), ('Support Vector Machine', SVC())]

training_results, names = [], []
test_results = []

build_and_evaluate_models()
test_models()

test_results.sort(key=lambda x: x[1], reverse=True)

print("Results from best to worst: ")
for n, v in test_results:
    print("Model: {}, Accuracy: {}".format(n, v))
