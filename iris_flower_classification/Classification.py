import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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
# Algorithms
models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC())]

results, names = [], []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))
