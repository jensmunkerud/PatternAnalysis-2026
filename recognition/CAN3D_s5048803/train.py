import dataset

dataPath = "./HipMRI_2D/semantic_MRs_anon"

x = dataset.load_data_2D("dataPath", normImage=True, early_stop=True)