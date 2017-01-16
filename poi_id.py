#!/usr/bin/python

import sys
import pickle
import numpy
import copy
import matplotlib.pyplot as plt
import pandas as pd


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from dict_to_csv import dict_to_csv
from algorithm import GetClf

print "\n###############START##################"
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'total_payments', 'long_term_incentive', 'expenses', 'exercised_stock_options', 'restricted_stock', 'from_messages', 'other','shared_receipt_with_poi', 'from_this_person_to_poi', 'from_poi_to_this_person_ratio2', 'from_this_person_to_poi_ratio1', 'from_this_person_to_poi_ratio2', 'to_messages','from_poi_to_this_person_ratio1' ,'bonus','from_poi_to_this_person', 'loan_advances']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#########################################################################################
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_dataset.pop('TOTAL')
############################################
# convert the enron data from dict format to csv file.
dict_to_csv(my_dataset, output_file='enron.csv' ,overwrite = True)
####################################################
### Task 3: Create the  new four features feature(s) [ read the report enron.ipynb for explaination]
# first create the two ratio features out of the total messages of the person
for person in my_dataset.keys():
	my_dataset[person]['from_this_person_to_poi_ratio1'] = round( float( my_dataset[person]['from_this_person_to_poi'] )/\
	 float( my_dataset[person]['to_messages'] ), 5)
	my_dataset[person]['from_poi_to_this_person_ratio1'] = round( float( my_dataset[person]['from_poi_to_this_person'] )/\
	 float( my_dataset[person]['from_messages'] ), 5)
#******************************************************
#create  from/to person/poi ratio 2 , which is based on sum of pois' 'from/to messages'
to_poi_messages   = 0
from_poi_messages = 0
for person in my_dataset.keys():
	if int(my_dataset[person]['poi']) == 1 :
		try :
			to_poi_messages   += int(my_dataset[person]['to_messages'])
		except ValueError :
			pass
		try :
			from_poi_messages += int(my_dataset[person]['from_messages'])
		except ValueError :
			pass
for person in my_dataset.keys():
	my_dataset[person]['from_this_person_to_poi_ratio2']   = round( float( my_dataset[person]['from_this_person_to_poi'] )/\
	 float(from_poi_messages), 5)
	my_dataset[person]['from_poi_to_this_person_ratio2']   = round( float( my_dataset[person]['from_poi_to_this_person'] )/\
	 float(from_poi_messages), 5)
################################################################################
#convert any nan values to NaN.
for person in my_dataset.keys():
	for feature in my_dataset[person].keys():
		if str(my_dataset[person][feature]) == 'nan':
			my_dataset[person][feature] = 'NaN'
################################################################
data               = featureFormat(my_dataset, features_list,  sort_keys = True)
labels, features   = targetFeatureSplit(data)
#########################################
## Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = copy.deepcopy(data)
for i in range(1, len(data[0]), 1) :
	result = scaler.fit_transform( numpy.array([ float(x) for x in data[:,i]]).reshape(len(data), 1) )
	data_scaled[:,i] = [ x[0] for x in result ]
labels, features_scaled = targetFeatureSplit(data_scaled)
###########################################################
## feature selection
from sklearn.feature_selection import SelectKBest
import operator
kbest = SelectKBest(k = 'all')
kbest.fit(features, labels)
scores = { k:v  for k,v in zip(features_list, kbest.scores_) }
scores = sorted(scores.items(), key=operator.itemgetter(1), reverse = True )
for k,v in scores:
	print '{}       {}'.format(k,v)
print "**********"
###########################################################
### split the data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_scaled, labels, test_size=0.3, random_state=32)

#######################################
### classifier training and predicting
#parameters :
#  			- algorithm = [ 'DT', 'svc', 'BDT', 'BSVC', 'RandF' ]
#  			- CV = [ True, False ] # whether to use GridSearchCV or not
#  			- labels : must be enterd if CV == 1
clf = GetClf('RandF',1, labels)
clf.fit(features, labels)
########################################################################
# Evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
pred = clf.predict(features_test)
print '*********************************************************'
print 'score_train: ', clf.score(features_train, labels_train)
print 'score_test: ' , clf.score(features_test, labels_test)
print 'precision: '  , precision_score(labels_test, pred)
print 'recall: '     , recall_score(labels_test, pred)
print 'F1: '         , f1_score(labels_test, pred)
print 'accuracy: '   , accuracy_score(labels_test, pred)
print  classification_report(labels_test, pred)
print '******************************************************'

# if applicable print feature importances sorted
try :
	importances = clf.feature_importances_
	print "*********feature**importances**********"
	for i in range(1,len(features_list)+1) :
		if importances[i-1] > 0 :
			print "** {0} : {1}".format(features_list[i], importances[i-1] )
	print "****************************************"

except :
	pass

# if applicable print GridSearchCV best_score and best_parameters
try :
	print 'GS score: {}'.format(clf.best_score_)
	print clf.best_params_
except :
	pass

# uncomment to run the test function from tester.py
#test_classifier(clf, my_dataset, features_list, 1000)

#########################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need  to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
print "########################END#####################"
