import matplotlib.pyplot as plt
from sklearn import datasets, metrics
import numpy as np
import pandas as pd
import seaborn as sns

iris = datasets.load_iris()

#iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
#iris_df['Species'] = np.array([iris.target_names[cls] for cls in iris.target])
#sns.pairplot(iris_df, hue='Species') #графики разбиение сортов по признакам
#plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target,
    test_size=0.3, stratify=iris.target, random_state=42
)
rf_model = random_forest.fit(x_train, y_train)
predictions = rf_model.predict(x_test)
print('Accuracy "Predictions": {:.2f}'.format(accuracy_score(y_test, predictions)))

confusion_scores =  confusion_matrix(y_test, predictions)
confusion_df = pd.DataFrame(confusion_scores, columns=iris.target_names, index=iris.target_names)
sns.heatmap(confusion_df, annot=True) #график попадания цветков по признакам
plt.show()

feature_importance = list(zip(iris.feature_names, rf_model.feature_importances_))
feature_importance_df = pd.DataFrame(feature_importance, columns=['Feature',
                                                                  'RF Importance'])
#print(feature_importance_df) #у какого признака наибольший вес
#rf_model.get_params() #параметры модели

from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42) #100 деревьев
gb_model = gradient_boosting.fit(x_train, y_train) #обучение модели (max_dept - глубина дерева
print('Accuracy "Gradient Boosting": {:.2f}'.format(gb_model.score(x_test, y_test)))

feature_importance_df['GB importance'] = gb_model.feature_importances_
print(feature_importance_df)