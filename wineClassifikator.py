import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# загружаем набор данных. Набор взят с сайта Kaggle.com

dataset = pd.read_csv('wine.csv')

#смотрим структуру набора данных. Выводим 5 первых записей
print(dataset.head(5))


# выводим количество пустых значений в каждом столбике
for column in dataset.columns:
    print(f'{column:20} {dataset.high_quality[dataset[column].isnull()].count()}')


# заменяем пустые значения на среднее значение по столбику
dataset.volatile_acidity.loc[dataset.volatile_acidity.isnull()] = dataset.volatile_acidity.mean()
dataset.residual_sugar.loc[dataset.residual_sugar.isnull()] = dataset.residual_sugar.mean()
dataset.free_sulfur_dioxide.loc[dataset.free_sulfur_dioxide.isnull()] = dataset.free_sulfur_dioxide.mean()
dataset.density.loc[dataset.density.isnull()] = dataset.density.mean()
dataset.sulphates.loc[dataset.sulphates.isnull()] = dataset.sulphates.mean()

for column in dataset.columns:
    print(f'{column:20} {dataset.high_quality[dataset[column].isnull()].count()}')


#выводим подробную информацию по каждому столбику
for column in dataset.columns[1:]:
    print(dataset[column].describe())
    print('\n ---------- \n')


# заменяем текстовое значения цвета вина на числовое. 0 - red. 1 - white
colors = {'red': 0, 'white' : 1}
dataset['color'] = dataset['color'].map(colors)


# разделение набора данных на метки и наборы признаков
labels = dataset['high_quality']
data = dataset.drop(['high_quality' , "number"], axis=1)


# приведение всех значений к одному диапозону
scaler = StandardScaler()
scaler.fit(data)
data_train = scaler.transform(data)

# разедение набора на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(data_train, labels, test_size=0.20, random_state = 1)


# визуализируем значения столбцов по отношение к меткам. за исключением столбца "color". так как вино высокого качества может быть любого цвета
plt.figure(figsize=(20,24))
plot_number = 0
for column in data.drop(['color'], axis=1).columns:
    plt.scatter(x=data[column], y=labels, marker='o', c='r', edgecolor='b')
    plot_number += 1
    plt.subplot(4, 3, plot_number)
    plt.title('')
    plt.xlabel(column)
    plt.ylabel('high quality')
plt.show()


# создание различнх моделей для определения лучшей
models = []
models.append(("LineR" , LinearRegression()))
models.append(('LogicR' , LogisticRegression(solver='liblinear' , multi_class='ovr')))
models.append(('KNN' , KNeighborsClassifier()))
models.append(('SVM' , SVC(gamma='auto')))
models.append(('RandomForest' , RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=8, min_samples_leaf=2)))

# создание метода перевода результатов моделей регресии в целые числа. для определения принадлежности результата к классу
def linear_scorer(estimator, x, y):
    scorer_predictors = estimator.predict(x)

    scorer_predictors[scorer_predictors > 0.5] = 1
    scorer_predictors[scorer_predictors <= 0.5] = 0

    return metrics.accuracy_score(y, scorer_predictors)

results = []
names = []
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

# перебор различных моделей и вывод результатов
for name, model in models:
    if name in ['LineR' , 'LogicR']:
        cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring=linear_scorer)
    else:
        cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

# визуализация результатов различных моделей
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# создание словаря различных гиперпараметров
alg_frst_model = RandomForestClassifier(random_state=1)
alg_frst_params = [{
    'n_estimators': [350,400],
    'min_samples_split': [6,8],
    'min_samples_leaf': [1,2]
}]

# поиск лучших гиперпараметров методом перебора
alg_frst_grid = GridSearchCV(alg_frst_model, alg_frst_params, cv=cv, refit=True, verbose=1, n_jobs=1)   # поиск лучших параметров для модели
alg_frst_grid.fit(X_train, y_train)
alg_frst_best = alg_frst_grid.best_estimator_
print("Accuracy (random forest auto): {} with params {}".format(alg_frst_grid.best_score_, alg_frst_grid.best_params_))

# обучение лучшей модели с лучшими гиперпараметрами
alg_test = alg_frst_best
alg_test.fit(X_train, y_train)

# получение предсказаний по тестовым данным
predictions = alg_test.predict(X_test)

# вывод результатов точности предсказаний
print (metrics.accuracy_score(y_test, predictions))
print (metrics.classification_report(y_test, predictions))

# визуализация результатов предсказания модели. Матрица ошибок
sns.heatmap(metrics.confusion_matrix(y_test,predictions),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix for RF', y=1.05, size=15)
plt.show()