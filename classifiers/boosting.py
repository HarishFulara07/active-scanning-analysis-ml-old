import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from aux.train_classifier import train_classifier_using_grid_search


def learn(training_data_infile, trained_model_outfile = None, display_metrics: bool = False, gs_verbose: int = 0,
          n_jobs = 1):
	"""
	Trains a Gradient Boosting classifier

	:param training_data_infile: Csv file containing training data (labeled)
								 • The last column should be training labels
								 • Csv file can contain header (line 1 is skipped)
								 • Use: machine_learning.aux.data_processing.create_training_dataset

	:param trained_model_outfile: where to save the model
	:param display_metrics: whether to print model metrics or not
	:param gs_verbose: verbosity of GridSearch
	:param n_jobs: GridSearch parallel jobs
	:return:
	"""

	# testing parameters
	params = {
		'learning_rate': np.logspace(-2, 1, 4),
		'n_estimators': [100, 200, 300],
		'max_depth': [3, 5, 10],
		'min_samples_split': [2, 5]
	}

	return train_classifier_using_grid_search(
		classifier_name = 'Gradient Boosting',
		classifier_object = GradientBoostingClassifier,
		gs_params = params,
		training_data_infile = training_data_infile,
		trained_model_outfile = trained_model_outfile,
		display_metrics = display_metrics,
		gs_verbose = gs_verbose,
		n_jobs = n_jobs,
		normalize = False
	)
