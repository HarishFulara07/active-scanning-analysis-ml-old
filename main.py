import logging
import sys
import classifiers.bagging as bagging_classifier
import classifiers.boosting as boosting_classifier
import classifiers.decision_tree as decision_tree_classifier
import classifiers.knn as knn_classifier
import classifiers.linear_svm as linear_svm_classifier
import classifiers.logistic_regression as log_reg_classifier
import classifiers.naive_bayes as naive_bayes_classifier
import classifiers.random_forest as random_forest_classifier
import classifiers.rbf_svm as rbf_svm_classifier
import classifiers.sgd as sgd_classifier

DATASET = "dataset_new/merged_files/complete_merged_data.csv"


class LogFile(object):
	"""File-like object to log text using the `logging` module."""

	def __init__(self, name=None):
		self.logger = logging.getLogger(name)

	def write(self, msg, level=logging.INFO):
		self.logger.log(level, msg)

	def flush(self):
		for handler in self.logger.handlers:
			handler.flush()


if __name__ == '__main__':
	logging.basicConfig(level=logging.DEBUG, filename='out.log')
	sys.stdout = LogFile('stdout')
	n_jobs = 2
	verbose = 0
	bagging_classifier.learn(DATASET, "models/bagging.pkl", True, verbose, n_jobs)
	boosting_classifier.learn(DATASET, "models/boosting.pkl", True, verbose, n_jobs)
	decision_tree_classifier.learn(DATASET, "models/decision_tree.pkl", True, verbose, n_jobs)
	knn_classifier.learn(DATASET, "models/knn.pkl", True, verbose, n_jobs)
	linear_svm_classifier.learn(DATASET, "models/linear_svm.pkl", True, verbose, n_jobs)
	log_reg_classifier.learn(DATASET, "models/logistic_regression.pkl", True, verbose, n_jobs)
	naive_bayes_classifier.learn(DATASET, "models/naive_bayes.pkl", True, verbose, n_jobs)
	random_forest_classifier.learn(DATASET, "models/random_forest.pkl", True, verbose, n_jobs)
	rbf_svm_classifier.learn(DATASET, "models/rbf_svm.pkl", True, verbose, n_jobs)
	sgd_classifier.learn(DATASET, "models/sgd.pkl", True, verbose, n_jobs)
