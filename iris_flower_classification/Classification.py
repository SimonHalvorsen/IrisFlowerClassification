import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

print("Training results mean and standard deviation: ")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)  # 10-fold cross validation
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))

# Make predictions on validation dataset

lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions_lr = lr.predict(X_test)
print("\nAccuracy of the logistic regression model: {}".format(accuracy_score(y_test, predictions_lr)))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions_lr))
print("\nClassification report: ")
print(classification_report(y_test, predictions_lr))
print()

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions_lda = lda.predict(X_test)
print("\nAccuracy of the linear discriminant analysis model: {}".format(accuracy_score(y_test, predictions_lda)))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions_lda))
print("\nClassification report: ")
print(classification_report(y_test, predictions_lda))
print()

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions_knn = knn.predict(X_test)
print("\nAccuracy of the KNN model: {}".format(accuracy_score(y_test, predictions_knn)))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions_knn))
print("\nClassification report: ")
print(classification_report(y_test, predictions_knn))
print()

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
predictions_dtc = dtc.predict(X_test)
print("\nAccuracy of the decision tree classifier model: {}".format(accuracy_score(y_test, predictions_dtc)))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions_dtc))
print("\nClassification report: ")
print(classification_report(y_test, predictions_dtc))
print()

gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictions_gnb = gnb.predict(X_test)
print("\nAccuracy of the gaussian naive bayes model: {}".format(accuracy_score(y_test, predictions_gnb)))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions_gnb))
print("\nClassification report: ")
print(classification_report(y_test, predictions_gnb))
print()

svm = SVC()
svm.fit(X_train, y_train)
predictions_svm = svm.predict(X_test)
print("\nAccuracy of the SVM model: {}".format(accuracy_score(y_test, predictions_svm)))
print("\nConfusion matrix: ")
print(confusion_matrix(y_test, predictions_svm))
print("\nClassification report: ")
print(classification_report(y_test, predictions_svm))


