import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

data_dir = '/home/xiaodi/security/ContrastiveMixup/dataset/security/ClaMP/dataset/ClaMP_Integrated-5184.csv'
data_csv = pd.read_csv(data_dir)
data_csv.packer_type = pd.factorize(data_csv.packer_type)[0]
data_y = data_csv.to_numpy()[:, -1].astype(np.int)
idx_pos = [i for i, x in enumerate(data_y) if x == 1]
idx_neg = [i for i, x in enumerate(data_y) if x == 0]
data_list = data_csv.values.tolist()
data_pos = np.array(random.sample(data_list, int(0.01 * len(idx_pos))))
print(len(data_pos))
data_neg = np.array(data_list)[idx_neg]
data_array = np.vstack((data_pos, data_neg))

data_x = data_array[:, :-1]
data_y = data_array[:, -1]

data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(data_x, data_y, test_size=0.20, random_state=42)

print(len(data_array))
print(len(data_train_x))
print(len(data_test_x))
print(len(data_train_y))
print(len(data_test_y))