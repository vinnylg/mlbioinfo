from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import sys
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd


def main(data):
	print('loading data...')
	x_data, y_data = load_svmlight_file(data)
	#split data
	print('spliting data...')
	x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.3, random_state=5)

	x_train = x_train.toarray()
	x_test = x_test.toarray()

	#classificadores
	#clf = Perceptron(n_jobs =-1)
	#clf = svm.SVC()
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 500, stop = 30000, num = 500)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10, 15, 20, 25, 30]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4, 6, 8, 10, 12, 14, 16]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]

	clf = RandomForestClassifier(random_state=42)

	#grid = GridSearchCV(clf,{'n_estimators': n_estimators,
    #                      'max_features': max_features,
    #                      'max_depth': max_depth,
    #                      'min_samples_split': min_samples_split,
    #                      'min_samples_leaf': min_samples_leaf,
    #                      'bootstrap': bootstrap}, n_jobs=-1,cv = 5,scoring = 'accuracy', verbose = 1)

	grid = RandomizedSearchCV(clf, {'n_estimators': n_estimators,
                          'max_features': max_features,
                          'max_depth': max_depth,
                          'min_samples_split': min_samples_split,
                          'min_samples_leaf': min_samples_leaf,
                          'bootstrap': bootstrap}, n_jobs=-1,cv=5 ,scoring = 'accuracy', verbose = 1)

	grid.fit(x_train,y_train)
	data = pd.DataFrame(grid.cv_results_)
	data.head()
	data = pd.DataFrame(grid.cv_results_)[['params','rank_test_score','mean_test_score']]
	data.head()
	print(grid.best_params_)
	
	#clf = DecisionTreeClassifier(random_state=0,max_depth=None, min_samples_split=2)
	#clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5, n_jobs=-1)
	#clf = AdaBoostClassifier(n_estimators=100)
	#scores = cross_val_score(clf, x_data, y_data, cv=5)
	#scores.mean()
	#print(scores)

	#for i in range (0, len(sample)):
	#	parameter = grid[i]
	#	clf.set_params(**parameter)
	#	clf.fit(x_train,y_train)
	#	# predicao do classificador
	#	y_pred = clf.predict(x_test)
#
#	#	score = accuracy_score(y_train, y_pred)
#	#	data = {"parameter":str(parameter), "score":score}
#
#	#	# mostra o resultado do classificador na base de teste
#	#	print ('Accuracy: ',  clf.score(x_test, y_test))
#	#	
#	#	# cria a matriz de confusao
#	#	cm = confusion_matrix(y_test, y_pred)
#	#	print (cm)
	#	print(classification_report(y_test, y_pred, labels=[0,1,2,3,4,5,6]))


if __name__ == "__main__":
	main(sys.argv[1]) #features
     