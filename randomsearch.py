import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from baseline import Trainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

my_trainer = Trainer(data='csv/train.csv')
my_trainer._preprocess()

# bigger search space
knn = KNeighborsClassifier()
k_nearest = list(range(0,50,5))

# define parameters whose value space needs to be searched
param_grid = {'n_neighbors': k_nearest}

random_search = RandomizedSearchCV( estimator=knn,
                                    param_distributions=param_grid,
                                    n_iter=10,
                                   # zehn zufällige möglichkeiten
                                    cv=10,
                                    scoring='accuracy',
                                    n_jobs=4,
                                    return_train_score=True,
                                    random_state=42)


# fit iris data (reload if yu modified those variables)
y = my_trainer.train_y
vec=TfidfVectorizer()
X = vec.fit_transform(my_trainer.train_X)

random_search.fit(X, y)

# inspect the results
pd.options.display.max_colwidth = 100 

df = pd.DataFrame.from_dict(random_search.cv_results_)
df = df.sort_values(by=["rank_test_score"])
df_relevant = df['rank_test_score', 'mean_test_score', 'params',  'mean_fit_time']

print(df)
print(df_relevant)
print(random_search.best_params_)
