import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import operator

boston = datasets.load_boston()
boston.keys()
#print(boston.DESCR[100:1300])
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
#print(boston_df.head())
#plt.figure(figsize=(6,4))
sns.displot(boston.target)
#plt.xlabel("Price (in thousands)")
#plt.ylabel('Count')
#plt.tight_layout()
#plt.show()

linear_regression = LinearRegression()
model = linear_regression.fit(boston.data, boston.target)
feature_weight_df = pd.DataFrame(list(zip(boston.feature_names, model.coef_)))
feature_weight_df.columns = ['Feature', 'Weight']
# print(feature_weight_df)
#first_predicted = sum(map(lambda pair: operator.mul(*pair),
 #                         zip(model.coef_, boston.data[0])))
# print(first_predicted + model.intercept_) # предсказание верно

predicted = model.predict(boston.data)
# print(predicted[:10])
prediction_ground_truth_df = pd.DataFrame(list(zip(predicted, boston.target)))
prediction_ground_truth_df.columns = ['Predicted', "Ground truth"]
# prediction_ground_truth_df.head()
#plt.figure(figsize=(6, 4))
#plt.scatter(predicted, boston.target)
#plt.xlabel('Predicted')
#plt.ylabel('Ground truth')
#plt.plot([0, 50], [0,50], color='red')
#plt.tight_layout()
#plt.show() #график разброса предсказанных и реальных значений

cancer = datasets.load_breast_cancer()
cancer.keys()
print(cancer.DESCR[:760])
cancer_df = pd.DataFrame(cancer.data)
cancer_df.columns = cancer.feature_names
#print(cancer_df.head())

from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
model = logistic_regression.fit(cancer.data, cancer.target)
print(model.coef_)
prediction = model.predict(cancer.data)
print(prediction[:20])
prediction = model.predict_proba(cancer.data)
print(prediction[:20])
print('Accuracy: {}'.format(model.score(cancer.data, cancer.target)) )
print(model.get_params())

