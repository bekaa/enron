#!/usr/bin/python

# this file is meant for exploring the features, and choosing the fittest one to work with.
# tasks :
#   1. It converts the pickle dictionary into a csv file, by the function : pickle_to_csv()
#		 so I can import that file to Rstudio and work with it there.
#   2. Now the R script will check correlations, plots, and data size and choose the fittest plots,
#		that R script is in folder './Rscripts/' with name 'explore_features.R'.
#		Make sure to open/run it after  completing step 1 above.
#	3. Now after we choosed our variables, time to check for outliers and remove them by the function :
#				outlier_detection_removal()
#	4. split featues and labels to training and testing sets using stratified KFold
import sys
import os.path


def dict_to_csv(my_dataset, output_file = 'data/dict.csv', overwrite = False):
	# convert the enron data from dict format to csv format.
	if not overwrite and os.path.isfile(output_file) :
		return
	first_time = True
	with open(output_file, 'wb') as f :
		for person in my_dataset.keys():
			for feature in my_dataset[ person ].keys() :
				if first_time :
					# print the keys as labels in the first row of the csv file.
					for feature in my_dataset[ person ].keys():
						f.write('%s,' % feature)
					f.write('\n')
					first_time = False
				item = my_dataset[person][feature]
				if item == 'NaN' : item = 'NA'
				f.write('%s,' % item)
			f.write('\n')
	return

