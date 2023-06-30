import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import normalize

def load_datasets(data_folder_):
    datasets = []
    for filename in os.listdir(data_folder_):
        file_path = os.path.join(data_folder_, filename)
        df = pd.read_csv(file_path)
        dataset = (df.values.tolist(), filename)
        datasets.append(dataset)
    return datasets

def normalize_dataset(data):
    data = np.array(data)
    normalized_data = normalize(data, norm='l2', axis=1)
    return normalized_data


