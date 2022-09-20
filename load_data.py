import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.utils import to_categorical

# #############################################################################
# Upload the Dataset
dataset = pd.read_csv("./dataset/security/GIDS_Merged.csv")

dataset.rename(columns={"CAN ID": "CAN_ID"}, inplace=True)
# Convert the Dataset Hex Value to Decimal
dataset[["CAN_ID", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]] = dataset[
    ["CAN_ID", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]
].apply(lambda x: x.astype(str).map(lambda x: int(x, base=16)))
# #############################################################################
# Delete the values in Label Column consists of Zero
dataset.drop(dataset.index[dataset["Label"] == "0"], inplace=True)
# #############################################################################
"""
# Metged all the CSV to a single Data Frame
path = r'./' # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
dataset = pd.concat(li, axis=0, ignore_index=True)
dataset.to_csv('./Toyota_Merged_COSE.csv', index=False)
"""
# #############################################################################
# Convert the Dataset Hex Value to Decimal
# dataset = dataset.apply(lambda x: x.astype(str).map(lambda x: int(x, base=16)))
# #############################################################################
# #############################################################################
# Percentage of Sample for Experimentation


# dataset1 = dataset
# ds1 = dataset1[dataset1.Label == "Benign"].sample(frac=0.5)

# dataset2 = dataset
# ds2 = dataset2[dataset2.Label == "DoS"].sample(frac=0.1)

# dataset3 = dataset
# ds3 = dataset3[dataset3.Label == "Fuzzy"].sample(frac=0.1)

# dataset4 = dataset
# ds4 = dataset4[dataset4.Label == "RPM"].sample(frac=0.1)

# dataset5 = dataset
# ds5 = dataset5[dataset5.Label == "Gear"].sample(frac=0.1)

# frames = [ds1, ds2, ds3, ds4, ds5]
# dataset = pd.concat(frames)


# #############################################################################
# #############################################################################
# Dron NaN value from Data Frame
dataset = dataset.dropna()
# Check if dataset has more nan values or not
dataset.isnull().sum()

# #############################################################################

# #############################################################################
# Print Shape of Dataset
print("Shape of Data:", dataset.shape)
print("Total Number of Labels: {}".format(dataset.shape[0]))
print(
    "Total Number Benign Class: {}".format(dataset[dataset.Label == "Benign"].shape[0])
)
print(
    "Total Number Flood Attack Class: {}".format(
        dataset[dataset.Label == "DoS"].shape[0]
    )
)
print(
    "Total Number Fuzzy Attack Class: {}".format(
        dataset[dataset.Label == "Fuzzing"].shape[0]
    )
)
print(
    "Total Number RPM Malfunction Anomaly Class: {}".format(
        dataset[dataset.Label == "RPM"].shape[0]
    )
)
print(
    "Total Number Gear Malfunction Anomaly Class: {}".format(
        dataset[dataset.Label == "Gear"].shape[0]
    )
)
# #############################################################################

# #############################################################################
# Dataset details
print(dataset.head(5))
print(dataset.columns.values)
print(dataset.info())
print(dataset.describe())


df_train, df_val, df_test = np.split(
    dataset.sample(frac=1), [int(0.8 * len(dataset)), int(0.9 * len(dataset))]
)

X_train, X_val, X_test = (
    df_train.iloc[:, 0:11],
    df_val.iloc[:, 0:11],
    df_test.iloc[:, 0:11],
)
y_train, y_val, y_test = df_train.iloc[:, 11], df_val.iloc[:, 11], df_test.iloc[:, 11]

print(X_train.info())
print(type(X_train))

print(y_train)


def encode_labels(input_y):
    encoder = LabelEncoder()
    output_y = encoder.fit_transform(input_y)
    output_y = to_categorical(output_y, num_classes=5)
    return output_y


y_train = encode_labels(y_train)
y_val = encode_labels(y_val)
y_test = encode_labels(y_test)
