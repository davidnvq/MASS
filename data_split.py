import numpy as np
import pandas as pd

indices = np.array([i for i in range(1500)])
np.random.seed(23031994)

size_Test = 300
size_Di = [100, 200, 400, 600, 1000, 1200]

def Di_indices_split():
	# Test split
	test_indices = np.random.choice(indices.shape[0], size = size_Test, replace=False)
	other_indices = [i for i in range(indices.shape[0]) if i not in test_indices]

	test_indices = indices[test_indices]; test_indices.sort()
	other_indices = indices[other_indices]; other_indices.sort()

	# D1 split 

	D1_indices = np.random.choice(other_indices.shape[0], size=size_Di[0], replace=False)
	rest_indices = [i for i in range(other_indices.shape[0]) if i not in D1_indices]

	D1_indices = other_indices[D1_indices]; D1_indices.sort()
	other_indices = other_indices[rest_indices]; other_indices.sort()

	# D2 split 
	D2_indices = np.random.choice(other_indices.shape[0], size=size_Di[1] - size_Di[0], replace=False)
	rest_indices = [i for i in range(other_indices.shape[0]) if i not in D2_indices]

	D2_indices = np.append(D1_indices, other_indices[D2_indices]); D2_indices.sort()
	other_indices = other_indices[rest_indices]; other_indices.sort()


	# D3 split 
	D3_indices = np.random.choice(other_indices.shape[0], size=size_Di[2] - size_Di[1], replace=False)
	rest_indices = [i for i in range(other_indices.shape[0]) if i not in D3_indices]

	D3_indices = np.append(D2_indices, other_indices[D3_indices]); D3_indices.sort()
	other_indices = other_indices[rest_indices]; other_indices.sort()


	# D4 split 
	D4_indices = np.random.choice(other_indices.shape[0], size=size_Di[3] - size_Di[2], replace=False)
	rest_indices = [i for i in range(other_indices.shape[0]) if i not in D4_indices]

	D4_indices = np.append(D3_indices, other_indices[D4_indices]); D4_indices.sort()
	other_indices = other_indices[rest_indices]; other_indices.sort()


	#D5 split
	D5_indices = np.random.choice(other_indices.shape[0], size=size_Di[4] - size_Di[3], replace=False)
	rest_indices = [i for i in range(other_indices.shape[0]) if i not in D5_indices]

	D5_indices = np.append(D4_indices, other_indices[D5_indices]); D5_indices.sort()
	other_indices = other_indices[rest_indices]; other_indices.sort()

	#D6 split
	D6_indices = np.random.choice(other_indices.shape[0], size=size_Di[5] - size_Di[4], replace=False)
	rest_indices = [i for i in range(other_indices.shape[0]) if i not in D6_indices]

	D6_indices = np.append(D5_indices, other_indices[D6_indices]); D6_indices.sort()
	other_indices = other_indices[rest_indices]; other_indices.sort()

	return test_indices, D1_indices, D2_indices, D3_indices, D4_indices, D5_indices, D6_indices

def export_test_data():
	Features = pd.read_csv("data/features_TFIDF.csv", header=None, sep=",", dtype=np.float32)
	print (Features.shape[0]) 
	Labels = pd.read_csv("data/labels.csv", header=None, sep=",", dtype= np.integer)
	print(Labels.shape[0])
	indices = Di_indices_split()
	test_indices = indices[0]
	D4i_indices = indices[1:]
	Test_Labels = Labels.iloc[test_indices, : ]
	Test_Labels.to_csv("data/D4i/test_labels.txt", sep=",", header=False, index=False, index_label=False)
	Test_Features = Features.iloc[test_indices, : ]
	Test_Features.to_csv("data/D4i/test_features.txt", sep=",", header=False, index=False, index_label=False)
	return

def export_D4_data():
	Features = pd.read_csv("data/features_TFIDF.csv", header=None, sep=",", dtype=np.float32)
	Labels = pd.read_csv("data/labels.csv", header=None, sep=",", dtype= np.integer)

	indices = Di_indices_split()
	D4i_indices = indices[1:]
	def export_D4i_data(D_index, Features, Labels, name):
		D4i_Features = Features.iloc[D_index, : ]
		D4i_Labels = Labels.iloc[D_index, : ]
		D4i_Features.to_csv("data/D4i/D4" + name + "_features.txt", sep=",", header=False, index=False, index_label=False)
		D4i_Labels.to_csv("data/D4i/D4" + name + "_labels.txt", sep=",", header=False, index=False, index_label=False)
		return

	export_D4i_data(D4i_indices[0], Features, Labels, "a")
	export_D4i_data(D4i_indices[1], Features, Labels, "b")
	export_D4i_data(D4i_indices[2], Features, Labels, "c")
	export_D4i_data(D4i_indices[3], Features, Labels, "d")
	export_D4i_data(D4i_indices[4], Features, Labels, "e")
	export_D4i_data(D4i_indices[5], Features, Labels, "f")
	return 

def text_split():
	text = []
	with open("data/text_data/text_hotel1493.txt", encoding="utf8") as f:
		text = f.readlines()
	indices = Di_indices_split()
	test_indices = indices[0]
	D4i_indices = indices[1:]

	def write_file(text, indices, filename):
		lines = [text[i] for i in indices.tolist()]
		print(lines[0])
		text_file = open("data/text_data/D4i_text_data/D4" + filename + "_text_data.txt", "w", encoding="utf8")
		for line in lines:
			text_file.write(line)
		text_file.close()
		return 

	write_file(text, D4i_indices[0], "a")
	write_file(text, D4i_indices[1], "b")
	write_file(text, D4i_indices[2], "c")
	write_file(text, D4i_indices[3], "d")
	write_file(text, D4i_indices[4], "e")
	write_file(text, D4i_indices[5], "f")
	
	
	return


if __name__ == "__main__":
	export_test_data()