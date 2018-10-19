import glob
import numpy as np


def get_merged_data(directory_path):
	"""
	Read all the file names present in the given directory
	"""
	files = glob.glob(directory_path + '/**/*.csv', recursive=True)
	merged_data = np.array([])
	for file in files:
		data = np.genfromtxt(file, delimiter="|", skip_header=True)
		data = np.nan_to_num(data)
		if not data.any():
			continue
		if data.ndim == 1:
			data.resize((1, data.shape[0]))
		if merged_data.any():
			merged_data = np.concatenate((merged_data, data))
		else:
			merged_data = data
	return merged_data


if __name__ == '__main__':
	causes = ["apsp", "bl", "ce", "lrssi", "pscan-a", "pscan-u", "pwr"]
	dataset_single_files_dir = "dataset_new/single_files"
	dataset_merged_files_dir = "dataset_new/merged_files"
	for cause in causes:
		merged_data = get_merged_data(dataset_single_files_dir + "/" + cause)
		np.savetxt(dataset_merged_files_dir + "/" + cause + ".csv", merged_data, delimiter=",")
