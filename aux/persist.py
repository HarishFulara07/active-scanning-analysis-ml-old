import os

from sklearn.externals import joblib


def save_model(model, filepath):
	joblib.dump(model, filepath)
	print('Model saved at {:s}'.format(filepath))


def load_model(filepath):
	if not os.path.exists(filepath):
		print('{:s} doesn\'t exist'.format(filepath))
		return
	return joblib.load(filepath)
