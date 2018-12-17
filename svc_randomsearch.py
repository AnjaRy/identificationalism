import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from baseline import Trainer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

my_trainer = Trainer(data='csv/train.csv')
my_trainer._preprocess()

# bigger search space
estimator = SVC()

# Parameters
C_value = list(range(0, 100, 2))
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
degree = list(range(0, 5))
gamma = list(range(0, 1000, 20))
coef = list(range(0, 1))
shrinking = ['True', 'False']
probability = ['True', 'False']
decision_function = ['ovo', 'ovr']


# define parameters whose value space needs to be searched
param_grid = {'C': C_value,
		'kernel': kernel,
		'degree': degree,
		'gamma': gamma,
		'coef0': coef,
#		'shrinking': shrinking,
#		'probability': probability,
		'decision_function_shape': decision_function}

random_search = RandomizedSearchCV(estimator=estimator,
                                    param_distributions=param_grid,
                                    n_iter=1,
                                    # zehn zufällige möglichkeiten
                                    cv=10,
                                    scoring='accuracy',
                                    n_jobs=3,
                                    return_train_score=True,)


# fit iris data (reload if yu modified those variables)
y = my_trainer.train_y
vec = TfidfVectorizer()
X = vec.fit_transform(my_trainer.train_X)

random_search.fit(X, y)

# inspect the results
pd.options.display.max_colwidth = 100 

data = pd.DataFrame.from_dict(random_search.cv_results_)
sorted_data = data.sort_values(by=["rank_test_score"])

print(data)
print(random_search.best_params_)
