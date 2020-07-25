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


##############################################################################################################################################
# Q2
# a.
# Training data:
X_train = np.array([[0],[0.5],[2]])
Y_train = np.array([-1.25,-0.6,-4.85])

# Test data:
X_test = np.array([[-1],[1],[3]])
Y_test = np.array([-5.2,-0.9,-13])


##############################################################################################################################################
# b.
# Fitting linear regression model:
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)

# Predicting the Y from X according to the linear regression model:
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)
print("********************************************************************************")
print("2.b.")
print("The predicted values for X_test: ", y_test_pred)

# Calculating the regular LS estimators:
W_1 = lr_model.coef_
W_0 = lr_model.intercept_
print("LS estimators:")
print("w0: ", W_0)
print("w1: ", W_1)


##############################################################################################################################################
# c.
# Calculating the mean squared errors for the train set and the test set:
regular_train_MSE = mean_squared_error(Y_train, y_train_pred)
regular_test_MSE = mean_squared_error(Y_test, y_test_pred)
print("********************************************************************************")
print("2.c.")
print("The MSE of the regression on the training data: ", regular_train_MSE)
print("The MSE of the regression on the test data: ", regular_test_MSE)
# we can conclude that the error for the training set is much smaller than the test error because we trained on the training set.
# In adddition, the error is very large because the samples set is only 3 points.


##############################################################################################################################################
# d.
def leastSquareEstimator(X, Y):
    
    # Adding a column of 1s:
    ones = np.ones((len(X), 1))
    new_X = np.append(ones, X, axis=1)

    # Calculating W_LS = (X^T * X)^-1 * X^T * Y:
    XTX_inv = np.linalg.inv((new_X.T).dot(new_X))
    LS = XTX_inv.dot(new_X.T).dot(Y)
    return LS

print("********************************************************************************")
print("2.d.")
print("The least square estimator:", leastSquareEstimator(X_train, Y_train))


##############################################################################################################################################
# e.
# Plotting the regression line:
x_line = np.arange(-3,5,0.1)
y_line = W_0 + x_line*W_1
plt.plot(x_line, y_line, color = 'black', linestyle = 'dashed')
plt.scatter(X_train, Y_train, marker = '*')
plt.scatter(X_test, Y_test, marker = 'o')
plt.xlabel('X axis')
plt.ylabel('Y axis')
fname='D:\ML\HW1\plot2e.png'
# plt.savefig(fname)
# Does it look like the regression fit the data? - NO!


##############################################################################################################################################
# f.
# If we would mark z, how can we write yi in a linear form?

Z_train = np.append(X_train, X_train**2, axis = 1)
Z_test = np.append(X_test, X_test**2, axis = 1)

# f2.
W = leastSquareEstimator(Z_train, Y_train)
y_train_pred = W[0] + W[1] * Z_train[:, 0] + W[2] * Z_train[:, 1]
print(y_train_pred)
y_test_pred = W[0] + W[1] * Z_test[:, 0] + W[2] * Z_test[:, 1]
print(y_test_pred)

# f3.
# Calculating the mean squared error:
print("********************************************************************************")
print("2.f.3.")
print("The MSE of the regression on the training data: ", mean_squared_error(Y_train, y_train_pred))
print("The MSE of the regression on the test data: ", mean_squared_error(Y_test, y_test_pred))

# f4.
# Plotting both regressions:
x_line = np.arange(-3,5,0.1)
y_line_1 = W_0 + x_line*W_1
y_line_2 = W[0] + x_line*W[1] + (x_line**2)*W[2]
plt.plot(x_line, y_line_1, color = 'black', linestyle = 'dashed')
plt.plot(x_line, y_line_2, color = 'red', linestyle = 'dashed')
plt.scatter(X_train, Y_train, marker = '*')
plt.scatter(X_test, Y_test, marker = 'o')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title("Polynomial Regression vs. Linear Regression")
fname='D:\ML\HW1\plot2f4.png'
# plt.savefig(fname)


##############################################################################################################################################
# g.
# Which assumption did not hold, thus making the linear regression to fail?




##############################################################################################################################################
##############################################################################################################################################


# Q3
# a.
# Loading data:
df = pd.read_csv('parkinsons_updrs_data.csv')


##############################################################################################################################################
# b.
# Plotting a bar graph of how many males and females are:
sex_index = df['sex'].value_counts()
plt.figure(figsize=(10,5))
sn.barplot(sex_index.index, sex_index.values, alpha=0.9)
positions = (0, 1)
labels = ("Male", "Female")
plt.xticks(positions, labels)
plt.title('How many of each gender?')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Sex', fontsize=12)
fname='D:\ML\HW1\plot3b.png'
# plt.savefig(fname)


##############################################################################################################################################
# c.
# Plotting an histogram of the distribution of ages:
df.hist(column = 'age')
fname='D:\ML\HW1\plot3c.png'
# plt.savefig(fname)


##############################################################################################################################################
# d.
# Plotting a bar graph of how much "motor_UPDRS" varies between sexes:
plt.ylim(0, 25)
mean_sex = df.groupby('sex')['motor_UPDRS'].mean()
sn.barplot(mean_sex.index, mean_sex, alpha = 0.9)
positions = (0, 1)
labels = ("Male", "Female")
plt.xticks(positions, labels)
plt.xlabel('Sex')
plt.ylabel('motor_UPDRS')
plt.title('Sex vs motor_UPDRS', y=1)
fname='D:\ML\HW1\plot3d.png'
# plt.savefig(fname)


##############################################################################################################################################
# e.
# A scatter plot of 6 explanatory variables and motor_UPDRS:
matrix = df[['motor_UPDRS','total_UPDRS','Jitter.Per','Shimmer.dB','Shimmer.APQ5','Shimmer.APQ11','NHR']]
pd.plotting.scatter_matrix(matrix, alpha=0.2)
fname='D:\ML\HW1\plot3e.png'
# plt.savefig(fname)


##############################################################################################################################################
# f.
# Calculating the LS estimator using the 6 explanatory variables:
lr_model.fit(df[['total_UPDRS','Jitter.Per','Shimmer.dB','Shimmer.APQ5','Shimmer.APQ11','NHR']], df['motor_UPDRS'])
coefs = lr_model.coef_
intercept = lr_model.intercept_
print("\n********************************************************************************")
print("3.f.")
print("The least square estimator using sklearn:")
print("Coefficients: ", coefs)
print("Intercept: ", intercept)


##############################################################################################################################################
# g.
print("\n********************************************************************************")
print("3.g.")
print("The least square estimator using our function:")
print(leastSquareEstimator(df[['total_UPDRS','Jitter.Per','Shimmer.dB','Shimmer.APQ5','Shimmer.APQ11','NHR']], df['motor_UPDRS']))




##############################################################################################################################################
##############################################################################################################################################


# Q5
##############################################################################################################################################
# a.
iris = datasets.load_iris()
X = iris.data
y = iris.target


##############################################################################################################################################
# b.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1000)


##############################################################################################################################################
# c. + d. + e.
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


##############################################################################################################################################
# f.
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
fname='D:\ML\HW1\plot5f1.png'
# plt.savefig(fname)
sn.heatmap(normalized_cm, annot=True, cmap="tab20b")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalized confusion matrix')
fname='D:\ML\HW1\plot5f2.png'
# plt.savefig(fname)