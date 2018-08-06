#!/usr/bin/env python3
'''
authors: Chris Kim and Raymond Sutrisno

'''
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.callbacks import ModelCheckpoint
from utils import *
from ast import literal_eval
import os
import h5py

def build_keras_sequential(activation='sigmoid', architecture=(12,4), reg=None):
	"""
	Inputs: 
		architecture - tuple specifying the nodes in each layer of the model; should begin with the number of input features that are fed into the model and end
					   with the number of classes the model intends to classify the images into
		reg - regularizer object; pass None to leave weights as is
	Output:
		model - Keras Sequential neural net model
	Purpose: 
		The method instantiates  the Keras Sequential model. Layers are added based on the passed architecture and defined by a sigmoid activation function. 
		Weights are regularized based on the passed regularizer. The method compiles the neural network using the adam optimizer. As dewetting classification
		is a mulitclass problem, categorical crossentropy is passed as the loss function.
	"""
	model = Sequential()
	input_dim = architecture[0]
	# Adds hidden layers to Keras neural network
	for units in architecture[1:]:
		layer = Dense(units, 
					  activation=activation,
					  input_dim=input_dim,
					  kernel_regularizer=reg)
		model.add(layer)
		input_dim = units
	# Compiles Keras neural network
	model.compile(optimizer='adam',
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])
	return model

def get_training_metrics(records, epoch):
	"""
	Input: 
		records - list of k History objects for each k-fold of a model
		epoch - int representing desired epoch of interest
	Output: 
		the mean training accuracy and standard deviation at the passed epoch across k-folds
	"""
	scores = []
	for record in records:
		acc = record['acc']
		scores.append(acc[epoch-1])
	return np.mean(scores), np.std(scores)

def get_validation_metrics(records, epoch):
	"""
	Input: 
		records - list of k History objects for each k-fold of a model
		epoch - int representing desired epoch of interest
	Output:
		the mean cross-validation accuracy and standard deviation at the passed epoch across k-folds
	"""
	scores = []
	for record in records:
		val_acc = record['val_acc']
		scores.append(val_acc[epoch-1])
	return np.mean(scores), np.std(scores)

def plot_learning_curve(title, data, labels, config, reg, epoch, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
	Inputs: 
		title - str representing title of learning curve figure
		data - 2D numpy array (n, features) representing n training example features
		labels - 1D numpy array (n,) representing n traing example labels 
		config - tuple specifying the nodes in each layer of the model; should begin with the number of input features that are fed into the model and end
				 with the number of classes the model intends to classify the images into
		reg - regularizer object; pass None to leave weights as is
		epoch - int representing epoch to at which model should be trained until
		train-sizes - 1D numpy array; each element corresponds to a fraction of the training examples that will be used for plotting a learning curve
	Outputs: 
		plt - pyplot representing the learning curve
	Purpose:
		The method plots the training and cross-validation accuracies for the model trained only a subset of the training examples. The plot can help qualify
		the degree of variance and overfitting present within the model by portraying the relationship between the two curves.
	"""
	plt.figure()
	plt.title(title)
	plt.xlabel("Training examples")
	plt.ylabel("Score")

	points_num = train_sizes.shape[0]
	train_scores_mean = np.zeros(points_num)
	train_scores_std = np.zeros(points_num)
	test_scores_mean = np.zeros(points_num)
	test_scores_std = np.zeros(points_num)
	
	for k, size in enumerate(train_sizes):
		estimator = build_keras_sequential(architecture=config, reg=reg)
		select = int(data.shape[0] * size)-1
		print('\nCALCULATING WITH',select,'EXAMPLES ...')
		data_subset = data[:select, :]
		labels_subset = labels[:select]
		records, f1scores = train_keras(estimator, data_subset, labels_subset, epoch, batch_size=1)
		train_scores_mean[k], train_scores_std[k] = get_training_metrics(records, epoch)
		test_scores_mean[k], test_scores_std[k] = get_validation_metrics(records, epoch)
	
	plt.grid()
	# Converts train sizes from percentages to number of training examples
	train_sizes = (train_sizes * data.shape[0]).astype(int)
	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1, color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="b")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="b", label="Cross-validation score")

	plt.legend(loc='lower right')
	return plt

def new_build_and_train(title, data, labels, config, reg, epoch_num, split, n_splits=10, batch_size=10):
	"""
	Inputs:
		title - 
		data - 2D numpy array (n, features) representing n training example features
		labels - 1D numpy array (n,) representing n traing example labels 
		config - tuple specifying the nodes in each layer of the model; should begin with the number of input features that are fed into the model and end
				 with the number of classes the model intends to classify the images into
		reg - regularizer object; pass None to leave weights as is
		epoch - int representing epoch to at which model should be trained until
		split - float representing the fraction of the entire dataset devoted for testing
		n_splits - int representing how many folds StratifiedKFold cross-validation should execute
		batch_size - int representing the number of samples per gradient update
	Outputs: 
		records - list of k History objects for k folds
	Purpose:
		The method partitions the input data and labels into k-folds, with k determined by the passed n_splits parameter. StratifiedKFold from sklearn library preserves
		the percentage of samples for each class and is used to ensure that each sample is present in the training and cross-validation subsets, especially given the
		relatively small training set size. History objects collected from each fold are appended to a list. Models are saved every 25 epochs.
	"""
	con_matrix_labels = sorted(np.unique(labels))
	class_num = len(con_matrix_labels)
	con_matrix = np.zeros(shape=(class_num,class_num))
	
	kfold = StratifiedKFold(n_splits=n_splits,random_state=0).split(data,labels)
	records = []

	for j,(train,test) in enumerate(kfold):
		# Weights are reinitialized to random values by reinstantiating the model
		model = build_keras_sequential(architecture=config, reg=reg)
		source = 'nnmodels_training_split='+str(split)+'/config='+str(config)+',reg='+title[title.find('.')+1:-4]
		fold= ',fold='+str(j+1)+'.hdf5'
		filepath = source+',epoch={epoch:02d}'+fold
		# model is saved every 25 epochs
		checkpoint = ModelCheckpoint(filepath, 
									 verbose=0,
									 save_best_only=False,
									 period=25)
		callbacks_list = [checkpoint]
		# Record is a History object: record.history['acc'] and record.history['val_acc']
		record = model.fit(data[train],
						   to_categorical(labels[train],class_num),
						   validation_data=(data[test],to_categorical(labels[test],class_num)),
						   epochs=epoch_num,
						   batch_size=batch_size,
						   callbacks = callbacks_list,
						   verbose=0)
		records.append(record.history)
	return records

def plot_epoch_curve(title, records, epoch_list):
	"""
	Inputs:
		title - string representing the title of the epoch curve plot
		records - list of k History objects for k folds
		epoch_list - list of epochs corresponding to each point on the epoch curve
	Outputs:
		metrics_by_model - a 2D numpy array that stores the mean and std training and cross validation scores at each epoch from the passed epoch_list
	Purpose:
		The method plots an epoch curve, specifically the mean training and cross validation scores across k folds at each epoch of interest, specified in the
		epoch_list parameter. The region std +/- mean is filled in. The epoch curve demonstrates at which epoch interval the model begins to overfit on the training
		examples.
	"""

	plt.figure()
	plt.title('Training and CV Accuracy by Epoch (NN)')
	plt.xlabel('Epochs')
	plt.ylabel('Score')
	points_num = len(epoch_list)
	train_scores_mean = np.zeros(points_num)
	train_scores_std = np.zeros(points_num)
	test_scores_mean = np.zeros(points_num)
	test_scores_std = np.zeros(points_num)

	# metrics_by_model is a 2D numpy array with shape (len(epoch_list),6); each row signifies an epoch from the passed epoch list and takes the format
	# [train_mean, train_std, test_mean, test_std, 'config.reg', epoch]
	metrics_by_model = [None]*6
	metrics_by_model = np.array([metrics_by_model] * len(epoch_list))
	
	for i, epoch in enumerate(epoch_list):
		train_scores_mean[i], train_scores_std[i] = get_training_metrics(records, epoch)
		test_scores_mean[i], test_scores_std[i] = get_validation_metrics(records, epoch)
		metrics_by_model[i] = [train_scores_mean[i], train_scores_std[i], test_scores_mean[i], test_scores_std[i], title[19:-4], epoch]

	plt.grid()
	plt.fill_between(epoch_list, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1, color="r")
	plt.fill_between(epoch_list, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="b")
	plt.plot(epoch_list, train_scores_mean, 'o-', color="r", label="Training score")
	plt.plot(epoch_list, test_scores_mean, 'o-', color="b", label="Cross-validation score")
	plt.legend(loc='lower right')
	plt.savefig(title)
	plt.close()

	return metrics_by_model

def new_gridsearch(data, labels, layer1, layer2, regs, abridged, split):
	"""
	Inputs:
		data - 2D numpy array (n, features) representing n training example features
		labels - 1D numpy array (n,) representing n training example labels 
		layer1 - list of ints representing the number of nodes in the first hidden layer
		layer2 - list of ints representing the number of nodes in the second hidden layer
		regs - list of 3 lists, first of which will either be an empty list or [None],
			   second of which will either be an empty list or list of desired L1 regularizers,
			   and third of which will either be an empty list or list of desired L2 regularizers
		abridged - Boolean, True signifies the input data and labels do not contain any examples that have been labeled as Class 0, False otherwise
		split - float representing the fraction of the entire dataset devoted for testing
	Purpose:
		Performs a gridsearch within the desired hyperparameter space determined by layer1, layer2, and regs. Epoch plots are saved for each configuration,
		and the metrics, such as the training and cross-validation accuracies, are stored for each configuration as an excel file.
	"""
	epoch = 800
	epoch_list = [25*x+25 for x in range(epoch//25)]

	input_num = data.shape[1]
	output_num = np.unique(labels).shape[0]	
	model_configurations = []

	for i in layer1:
		for j in layer2:
			model_configurations.append((input_num, i, j, output_num))
		
	model_count=0
	reg_grp_dict = {0:'none', 1:'l1', 2:'l2'}
	penalty_dict = {-1:'',0:'-4',1:'-3'}
	penalty = -1

	total = len(model_configurations) * (len(regs[0])+len(regs[1])+len(regs[2]))
	dataset = np.array([[None]*6] * (total*len(epoch_list)))

	for config in model_configurations:
		for k, reg_group in enumerate(regs): # each 'reg_group' is a group of regularizers
			for j, reg in enumerate(reg_group): # each 'reg' is a regularizer within each group (None, L1, L2) 
				if reg != None:
					penalty = j
				else:
					penalty = -1
				if abridged:
					title = 'epochplots_woClass0/'+str(config)+'.'+reg_grp_dict[k]+penalty_dict[penalty]+'.png'
				else:
					title = 'epochplots_wClass0/'+str(config)+'.'+reg_grp_dict[k]+penalty_dict[penalty]+'.png'

				print('\nTraining and Cross-Validating Model',model_count+1,'of',total,'...')
				records = new_build_and_train(title, data, labels, config, reg, epoch, split)
				
				
				index = model_count * len(epoch_list)
				dataset[index : index + len(epoch_list)] = plot_epoch_curve(title, records, epoch_list)
				model_count += 1	
			
		# Output is saved every configuration change (i.e. every 5 model configurations)

		# Creates a dataframe of the np.array containing all the metrics and parameters
		df = pd.DataFrame(dataset)
		df = df.rename({0:'Mean_Train',1:'Std_Train',2:'Mean_Test',3:'Std_Test', 4:'Parameters',5:'Epoch'}, axis = 'columns')
		df = df.sort_values(by=['Mean_Test'], ascending = False)
	
		# Outputs dataframe to an Excel file
		title = file[file.find('_')+1:file.find('.')]
		writer = pd.ExcelWriter('data/gridsearch_'+title+'_NN_split='+split+'.xlsx') # file name
		df.to_excel(writer,title)
		writer.close()

def test_optimal_model(data_train, labels_train, data_test, labels_test, config, reg, reg_str, epoch, split):
	"""
	Inputs:
		data_train - 2D numpy array (n, features) representing n training example features
		labels_train - 2D numpy array (n, features) representing n training example features
		data_test - 2D numpy array (n, features) representing n testing example features
		labels_test - 2D numpy array (n, features) representing n testing example features
		config - tuple specifying the nodes in each layer of the model; should begin with the number of input features that are fed into the model and end
				 with the number of classes the model intends to classify the images into
		reg - regularizer object; pass None to leave weights as is
		reg_str - string describing what regularizer was passed
		epoch - int describing at which epoch the model should be trained until
		split - float representing the fraction of the entire dataset devoted for testing
	Outputs:
		score - float representing the testing accuracy for the model with the passed configuration parameters
		con_matrix - a 2D numpy array with shape (no. of classes, no. of classes) representing the confusion matrix for the model
	Purpose:
		The method trains a Keras Sequential model using the given configuration on the entire training set. The trained model is evaluated on the discrete testing set.
	"""
	con_matrix_labels = sorted(np.unique(labels_train))
	class_num = len(con_matrix_labels)
	con_matrix = np.zeros(shape=(class_num,class_num))

	model = build_keras_sequential(architecture=config, reg=reg)
	model.fit(data_train,
			  to_categorical(labels_train,class_num),
			  epochs=epoch,
			  batch_size=10,
			  verbose=0)
	print('Testing Neural Network with Optimal Hyperparameters (Architecture @',str(config),', Epochs @',epoch)
	score = model.evaluate(data_test,to_categorical(labels_test,class_num),batch_size=10,verbose=0) # score = [test loss, test accuracy]
	score = score[1]

	filepath = 'nnmodels_testing_split='+str(split)+'/acc='+str(score)+',config='+str(config)+',reg='+reg_str+',epoch='+str(epoch)+'.hdf5'
	model.save(filepath)

	predictions = model.predict_classes(data_test)
	con_matrix = con_matrix + confusion_matrix(labels_test,predictions,labels=con_matrix_labels)
	return score, con_matrix

def test_peak_epochs(data_train, labels_train, data_test, labels_test, abridged, split):
	"""
	Inputs:
		data_train - 2D numpy array (n, features) representing n training example features
		labels_train - 2D numpy array (n, features) representing n training example features
		data_test - 2D numpy array (n, features) representing n testing example features
		labels_test - 2D numpy array (n, features) representing n testing example features
		abridged - True signifies the input data and labels do not contain any examples that have been labeled as Class 0, False otherwise
		split - float representing the fraction of the entire dataset devoted for testing
	Purpose:
		The method tests all model configurations specified in the names of the epoch plots within the specified directory. An excel file is returned
		summarizing the results. Each row corresponds to a model configuration and contains, in order, the following:
		|Architecture|Regularizer|Epoch|Testing Accuracy|CV Accuracy, Mean|CV Accuracy, Std|Training Accuracy, Mean|Training Accuracy, Std| ... 
		... |Confusion Matrix, Row 0|Confusion Matrix, Row 1|Confusion Matrix, Row 2|Confusion Matrix, Row 3|f1 Score, Category 0||f1 Score, Category 1| ...
		... |f1 Score, Category 2|f1 Score, Category 3|
	"""
	if abridged is True:
		path = 'epochplots_woClass0/k='+str(split)
	else:
		path = 'epochplots_wClass0/k='+str(split)
	dirs = os.listdir(path)
	outputs = []

	inputs = data_train.shape[1]
	classes = np.unique(labels_train).shape[0]	

	for file in dirs:
		# Changes config type from string to tuple
		config = literal_eval(file[:file.find('.')])
		reg_str = file[file.find('.')+1:file.find('_')]
		if reg_str == 'none':
			reg = None
		else:
			if reg_str[:2] == 'l1':
				reg = regularizers.l1(10**int(reg_str[-2:]))
			elif reg_str[:2] == 'l2':
				reg = regularizers.l2(10**int(reg_str[-2:]))
			else:
				assert(False), 'Regularizer not identified'
		epoch = file[file.find('_')+1:]
		epoch = epoch[:epoch.find('.')]
		if epoch.find(',') == -1:
			epoch = [epoch]
		else:
			epoch = epoch.split(',')
		for e in epoch:
			e = int(e)
			acc, con_matrix = test_optimal_model(data_train, labels_train, data_test, labels_test, config, reg, reg_str, e, split)
			outputs.append([config, reg_str, e, acc, con_matrix])
			
	path = 'data/gridsearch_dewetting_woclass1_NN_k='+str(split)+'.xlsx'
	df = pd.read_excel(path)

	dataset = [None]*16
	dataset = np.array([dataset]*(len(dirs)*2))

	for k,output in enumerate(outputs):
		config = str(output[0])
		reg_str = output[1]
		epoch = output[2]
		test_acc = output[3]
		con_matrix = output[4]
		f1scores = confusion_matrix_f1_scores(con_matrix)
		c = []
		for i, row in enumerate(con_matrix):
			c.append(str((int(row[0]),int(row[1]),int(row[2]),int(row[3]))))
		entry = df[(df['Parameters']==(config+'.'+reg_str)) & (df['Epoch']==epoch)]
		if len(entry.index) == 0:
			assert(False), config+'.'+reg_str+str(epoch)
		mean_cv = entry.get_value(index=entry.index[0], col='Mean_Test')
		mean_train = entry.get_value(index=entry.index[0], col='Mean_Train')
		std_cv = entry.get_value(index=entry.index[0], col='Std_Test')
		std_train = entry.get_value(index=entry.index[0], col='Std_Train')
		dataset[k] = np.concatenate((np.array([config, reg_str, int(epoch), float(test_acc), float(mean_cv), float(std_cv), float(mean_train), float(std_train)]), np.array(c), f1scores), axis=0) #config, reg, epoch, test_acc, c0, c1, c2, c3, fs0, fs1, fs2, fs3
	
	# Creates a dataframe of the np.array containing all the metrics and parameters
	df = pd.DataFrame(dataset)
	df = df.rename({0:'Config',1:'Reg',2:'Epoch',3:'Test Acc', 4:'CV_Mean', 5:'CV_Std', 6:'Train_Mean', 7:'Train_Std', 8:'CM,Class0', 9:'CM,Class1', 10:'CM,Class2', 11:'CM,Class3', 12:'f1,0', 13:'f1,1', 14:'f1,2', 15:'f1,3'}, axis = 'columns')
	df = df.sort_values(by=['Test Acc'], ascending = False)
	
	# Outputs dataframe to an Excel file
	writer = pd.ExcelWriter('data/testing_dewetting_woclass1_NN_k='+str(split)+'_condensed.xlsx') # file name
	df.to_excel(writer,'condensed')
	writer.close()

def Main(file, cpu, abridged, split, task):
	metric_count = 6
	seed = 0
	# Sets seed for reproducibility
	np.random.seed(seed)
	
	testable = True

	nodes1 = [10,8,5]
	nodes2 = [8,5]
	regs = [[None],
			[regularizers.l1(0.0001), regularizers.l1(0.001)],
			[regularizers.l2(0.0001), regularizers.l2(0.001)]]

	if cpu == 0:
		nodes1 = [10]
	elif cpu == 1:
		nodes1 = [8]
	elif cpu == 2:
		nodes1 = [5]
	elif cpu == 3:
		
	elif cpu == 4:
		nodes1 = [8]
		nodes2 = [5]
		regs = [[None],
				[],
				[]]
	else:
		testable = False

	assert(testable==True), 'Current CPU has no jobs to process'

	data, labels = pre_process_data(file, pickled=False, feature_cols=[], label_col=-1, drop=['file_names'],
	                                one_hot=False, shuffle=True, standard_scale=True, index_col=0)
	data_train, data_test, labels_train, labels_test = train_test_split(data,labels,test_size=split,random_state=seed)

	# If class 0 exists, moves an class 0-labeled examples from the training set to the testing set
	# The if-statement is hard-coded for the given file and seed
	if abridged is not True:
		labels_test = np.append(labels_test, labels_train[-6])
		data_test = np.append(data_test, [data_train[-6]], axis = 0)
		labels_train = np.delete(labels_train,-6)
		data_train = np.delete(data_train,-6,axis=0)

	if task == 0:
		new_gridsearch(data_train, labels_train, nodes1, nodes2, regs, abridged, split)
	elif task == 1:
		test_peak_epochs(data_train, labels_train, data_test, labels_test, abridged, split)
	else:
		assert(False), 'Invalid or no task entered'

if __name__=='__main__' and len(sys.argv) > 1:
	"""
	To run script, run the following:
		python <csv file of features> -cpu <int> -split <float> -task <int>

		If csv file does not contain any examples labeled as Class 0, must be specified within the file name
			e.g. features_dewetting_woclass0,1.csv

		10% of the dataset is, by default, reserved for the testing set. Passing an argument after -split will change the fraction of the dataset that is reserved for testing.

		Gridsearch and testing cannot be run concurrently, as manual labeling of the peak cross-validation scores must be made from the epoch plots generated by gridsearch.
		An argument must be passed after -task to signify which task is desired. Passing nothing will raise an AssertionError.
	"""
	import argparse
	parser = argparse.ArgumentParser(description="""This python script performs GridSearch on various parameters for Neural Networks on data sets.""")
	parser.add_argument('data_set', type=str, nargs='*',default=[],\
						help=\
						"""
						The data set in csv form. This script assumes the first column is the
						index column and that there is a column called file_names which it will drop
						before reading.""")
	parser.add_argument('-cpu', type=int, action='store', default=0,\
						help=
						""" 
						Splits the gridsearch task to different CPU's
						0 = Chris
						1 = Ray
						2 = Kurt
						3 = Other
						""", dest="CPU")
	parser.add_argument('-split', type=float, action='store', default=0.1,\
						help=
						"""
						Determines the training and testing split
						Input of 0.1 will split the dataset into training set (90%) and testing set (10%)
						""", dest='SPLIT')
	parser.add_argument('-task', type=int, action='store',default=-1,\
						help=
						"""
						0 = gridsearch
						1 = testing
						""", dest="TASK")

	arguments = parser.parse_args()

	for file in arguments.data_set:
		file_nm = '{}/{}'.format('data', file)
		cpu = arguments.CPU
		split = arguments.SPLIT
		task = arguments.TASK

		if file_nm.find('0') > 0: # if abridged, no need to put class 0 into testing set
			Main(file_nm, cpu, True, split, task)

		else:
			Main(file_nm, cpu, False, split, task)