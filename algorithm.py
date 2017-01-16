#!/usr/bin/env python
'''
----------------------------------------------------------------------------------------
Implementation for the following algorithms :
	- Decision Tree 				>> DT()
	- Support Vector Machine 		>> svc()
	- Adaboost with Decision Trees  >> BDT()
	- Adaboost with svm             >> BSVC()
	- Random Forest                 >> RandF()

* To use this file, call the main fucntion GetClf():

  - parameters :
  			- algorithm = [ 'DT', 'svc', 'BDT', 'BSVC', 'RandF' ]
  			- CV = [ True, False ] # whether to use GridSearchCV or not
  			- labels : must be enterd if CV == 1

  - return :
  			- if CV == True :  GridSearchCV classifier object
  			- if CV == False : classifier object for the calling algorithm
----------------------------------------------------------------------------------------
'''

import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  StratifiedShuffleSplit

############
#   MAIN   #
# -------- #
# FUNCTION #
############
def GetClf(algorithm = 'DT', CV = False, labels = None):
	#clf = DecisionTree(features_train, labels_train)
	if algorithm == 'DT':
		return DecisionTree(CV,labels)
	elif algorithm == 'SVC':
		return svc(CV,labels)
	elif algorithm == 'BSVC':
		return BSVC(CV,labels)
	elif algorithm == 'BDT':
		return BDT(CV,labels)
	elif algorithm == 'RandF':
		return RandF(CV,labels)
	else :
		print 'wrong input parameter'
		return

##################################################################
def DecisionTree(CV, labels):
	params ={
	'min_samples_split': range(2,12,5),
	'max_features' : ('sqrt', 'log2')
	}
	DT = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 2, max_features = 'sqrt', random_state = 42)
	if CV :
		sss = StratifiedShuffleSplit(n_splits = 100, test_size = .3, random_state = 42)
		CV = GridSearchCV(DT, params, n_jobs = 3, scoring="f1", cv=sss)
		return CV
	return DT
#############################################
def svc(CV, labels) :
	params = {
	'C'     : [ 2**x for x in numpy.arange(0,04,.25) ],
	'gamma' : [ 2**x for x in numpy.arange(7,11,.25) ]
	}
	svc = SVC(kernel = 'rbf', C = 1, gamma = 128, random_state=42)
	if CV :
		sss = StratifiedShuffleSplit(n_splits = 100, test_size = .3, random_state = 42)
		CV = GridSearchCV(svc, params,  n_jobs = 3, scoring="f1", cv=sss)
		return CV
	return svc
######################
def BSVC(CV, labels):
	params = {
	'n_estimators' : range(8,20,1),
	'algorithm' : ('SAMME.R','SAMME'),
	'learning_rate': numpy.arange(.1,2,.1)
	}
	svc = SVC(kernel = 'rbf', C = 1, gamma = 128, probability=True, random_state=42)
	ada = AdaBoostClassifier( svc, n_estimators = 8, algorithm='SAMME.R', learning_rate = .001 )
	if CV :
		sss = StratifiedShuffleSplit(n_splits = 100, test_size = .3, random_state = 42)
		CV = GridSearchCV(ada, params, n_jobs = -1, scoring="f1", cv=sss)
		return CV
	return ada
#################################
def BDT(CV, labels):

	params = {
	'n_estimators' : range(10,15,1),
	'learning_rate': numpy.arange(.001,1,.05),
	'algorithm' : [ 'SAMME', 'SAMME.R' ]
	}
	DT = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 7, max_features = 'sqrt', random_state = 42)
	ada = AdaBoostClassifier(DT, n_estimators = 13, learning_rate = 0.35, algorithm = 'SAMME')
	if CV :
		sss = StratifiedShuffleSplit(n_splits = 100, test_size = .3, random_state = 42)
		CV = GridSearchCV(ada, params, n_jobs = -1, scoring="f1", cv=sss)
		return CV
	return ada
####################################################
def RandF(CV, labels):
	params  = {
	#'max_features' : [ 'sqrt', 'auto', 'log2'],
	'min_samples_split' : range(2,10),
	'n_estimators' : range(6,14)
	}
	RF = RandomForestClassifier( max_features='sqrt', min_samples_split=2, n_estimators=9, oob_score=False, random_state=42, n_jobs=-1 )
	if CV :
		sss = StratifiedShuffleSplit(n_splits = 100, test_size = .3, random_state = 42)
		CV = GridSearchCV(RF, params,  n_jobs = -1, scoring="f1", cv=sss)
		return CV
	return RF
############################################################
