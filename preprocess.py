import os
import numpy as np


def get_processed_csv_file_names(directory_path):
	"""
	Read all the file names present in the given directory
	"""
	__supported_extensions = ['.csv', ]

	processed_csv_file_names = list()

	listdir = os.listdir(directory_path)
	for file in listdir:
		if os.path.splitext(file)[1] in __supported_extensions:
			processed_csv_file_names.append(file)

	# sort so that we always read in a predefined order
	# key: smallest file first
	processed_csv_file_names.sort(key = lambda f: os.path.getsize(os.path.join(directory_path, f)))
	return processed_csv_file_names


if __name__ == '__main__':
	dataset_path = "dataset_new/merged_files"
	file_names = get_processed_csv_file_names(dataset_path)
	complete_data = np.array([])
	for label, file_name in enumerate(file_names):
		data = np.genfromtxt(dataset_path + "/" + file_name, delimiter=",", skip_header=False)
		labels = np.ones((data.shape[0], 1)) * label
		data = np.concatenate((data, labels), axis=1)
		if complete_data.any():
			complete_data = np.concatenate((complete_data, data))
		else:
			complete_data = data
	print(complete_data.shape)
	np.savetxt(dataset_path + "/complete_merged_data.csv", complete_data, delimiter=",")
