import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from baseline import Trainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

my_trainer = Trainer(data='csv/train.csv')
my_trainer._preprocess()

# bigger search space
knn = KNeighborsClassifier()
k_nearest = list(range(5,100))

# define parameters whose value space needs to be searched
param_grid = {'n_neighbors': k_nearest}

random_search = RandomizedSearchCV( estimator=knn,
                                    param_distributions=param_grid,
                                    n_iter=40,
                                   # zehn zufällige möglichkeiten
                                    cv=10,
                                    scoring='accuracy',
                                    n_jobs=3,
                                    return_train_score=True,)


# fit iris data (reload if yu modified those variables)
y = my_trainer.train_y
vec=TfidfVectorizer()
X = vec.fit_transform(my_trainer.train_X)

random_search.fit(X, y)

# inspect the results
pd.options.display.max_colwidth = 100 

data = pd.DataFrame.from_dict(random_search.cv_results_)
data_sorted = data.sort_values(by=["rank_test_score"])

print(data_sorted)
print(random_search.best_params_)

