import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from baseline import Trainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize


def randomsearch(max):
	my_trainer = Trainer(data='csv/train.csv')
	my_trainer._preprocess()

	# bigger search space
	knn = KNeighborsClassifier()
	k_nearest = list(range(30,33))

	# define parameters whose value space needs to be searched
	param_grid = {'n_neighbors': k_nearest}

	random_search = RandomizedSearchCV( estimator=knn,
        	                            param_distributions=param_grid,
                	                    n_iter=3,
                        	           # zehn zufällige möglichkeiten
                                	    cv=10,
                                    	scoring='accuracy',
                                    	n_jobs=3,
                                    	return_train_score=True,)


	# fit iris data (reload if yu modified those variables)
	y = my_trainer.train_y
	vec=TfidfVectorizer(strip_accents='ascii', preprocessor=None, tokenizer=word_tokenize, max_df=max)
	X = vec.fit_transform(my_trainer.train_X)

	random_search.fit(X, y)

	pd.options.display.max_colwidth = 100 

	data = pd.DataFrame.from_dict(random_search.cv_results_)
	data_sort = data.sort_values(by=["rank_test_score"])
	data_relevant = data_sort[['rank_test_score', 'mean_test_score', 'params',  'mean_fit_time']]

	print(data)
	print(data_relevant)
	print(random_search.best_params_)
	print('NEXT')

if __name__ == '__main__': 
	for i in np.linspace(0.7,1,7):
		print(i)
		randomsearch(i)


#stop_words=stopwords.words("english")
