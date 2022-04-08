import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


#cancer = datasets.load_breast_cancer()
#logistic_regression = LogisticRegression()
#model = logistic_regression.fit(cancer.data, cancer.target) #обучение логистической регрессии
#print('Accuracy: {:.2f}'.format(model.score(cancer.data, cancer.target)))

#prediction = model.predict(cancer.data) #обучение модели на данных
#print('Accuracy: {:.2f}'.format(metrics.accuracy_score(cancer.target, prediction))) #точность предсказания
#print('ROC AUC: {:.2f}'.format(metrics.roc_auc_score(cancer.target, prediction)))
#print('F1:{:.2f}'.format(metrics.f1_score(cancer.target, prediction)))

from sklearn.model_selection import train_test_split

# cancer.data - признаки, cancer.target -целевая переменная
#x_train, x_test, y_train, y_test = train_test_split(
#    cancer.data, cancer.target, test_size= 0.2, random_state= 12)
#model = logistic_regression.fit(x_train, y_train)

#сравнение модели на тренировочной и тестовой части
#print("Train accuracy: {:.2f}".format(model.score(x_train, y_train)))
#print("Test accuracy: {:.2f}".format(model.score(x_test, y_test)))

#from sklearn.linear_model import Lasso, Ridge, ElasticNet
#boston = datasets.load_boston
# разные виды регуляризации
#lasso = Lasso()
#ridge = Ridge()
#elastic = ElasticNet()

#for model in [lasso, ridge, elastic]:
#    x_train, x_test, y_train, y_test = train_test_split(
#        cancer.data, cancer.target, test_size=0.2)
#    model.fit(x_train, y_train)
#    predictions = model.predict(x_test)
 #   print(model.__class__)
  #  print('MSE:{:.2f}\n'.format(metrics.mean_squared_error(y_test, predictions)))
 #

    # коэффиценты детерминации
#print('R2: {:.2f}'.format(model.score(x_test, y_test)))
#print('R2: {:.2f}'.format(metrics.r2_score(y_test, predictions)))

# Кросс-валидация
from sklearn.model_selection import KFold, cross_val_score

iris = datasets.load_iris()
#print(iris.keys())
#print((iris.DESCR[:475]))
#print(iris.target)
logistic_regression = LogisticRegression()
cv = KFold(n_splits=5) #Stratified KFold

for split_idx, (train_idx, test_idx) in enumerate(cv.split(iris.data)):

