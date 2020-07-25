from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn


iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1000)

SETOSA = 0
VERSICOLOR = 1
VIRGINCACV = 2
NUMBER_OF_TYPES = 3

# Classifying the three irises:
def build_set(data, iris_type):
    return [1 if current_type == iris_type else -1 for current_type in data]


def get_training_and_test(iris_type):
    return build_set(y_train, iris_type), build_set(y_test, iris_type)


def predictor(iris_type):
    y_train_set, y_test_set = get_training_and_test(iris_type)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train_set)
    return log_reg.predict_proba(X_test)[:, 1]


classify_setosa = predictor(SETOSA)
classify_versicolor = predictor(VERSICOLOR)
classify_virgincacv = predictor(VIRGINCACV)

print("\n********************************************************************************")
print(list(zip(classify_setosa, classify_versicolor, classify_virgincacv)))


def classify_between(classifier1, classifier2, classifier3):
    return [np.argmax(predictions) for predictions in zip(classifier1, classifier2, classifier3)]


y_pred = classify_between(*[list(predictor(iris_type)) for iris_type in range(NUMBER_OF_TYPES)])
print(y_pred)
print(y_test)



# Plotting the confusion matrices:
plt.clf()
plt.cla()
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float')
normalized_cm = cm / cm.sum(axis=1)[:, np.newaxis]
sn.heatmap(cm, annot=True, cmap="tab20b")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix')
fname=''
# plt.savefig(fname)
sn.heatmap(normalized_cm, annot=True, cmap="tab20b")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalized confusion matrix')
fname=''
# plt.savefig(fname)