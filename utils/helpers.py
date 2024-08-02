import pandas as pd
import pickle
import random
import math
from rdkit import Chem
from mordred import Calculator, descriptors

def get_descriptors(smiles):
    mols = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        mols.append(mol)
        
    calc = Calculator(descriptors, ignore_3D=True)
    all_descriptors = calc.pandas(mols)
    return all_descriptors

# Load in data from pickle files and save to excel file
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    return df

# load data from a csv file
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# write data to a csv file
def write_csv(df, file_path):
    df.to_csv(file_path, index=False)

# load data from excel
def load_excel(file_path):
    df = pd.read_excel(file_path)
    return df

# take the size of a dataframe and return a list for the train set indices based on the size and percent
def train_indices(df_size, percent):
    train_size = int(df_size * percent)
    train_indices = random.sample(range(df_size), train_size)
    return train_indices

# take the df and the list of the train indices. Slipt in train and val set
def train_val_split(df, train_indices):
    train = df.iloc[train_indices]
    val = df.drop(train_indices)
    return train, val

#randomly pick the training set
def random_sampling(features, label, train_set_size = 342, indices = None):
    if indices is None:
        indices = list(range(features.shape[0]))
    random.shuffle(indices)
    train_indices = indices[:train_set_size]
    test_indices = indices[train_set_size:]
    features_train, label_train = features.loc[train_indices], label.loc[train_indices]
    features_test, label_test = features.loc[test_indices], label.loc[test_indices]
    return features_train, label_train, features_test, label_test

# dvide range of label[2] in num_bins and then take perecent_from_bin from each bin to form the training set. Put the rest into the testing set
# return 4 dataframes: features_train, label_train, features_test, label_test
def stratified_sampling(df_features, labels, num_bins, perecent_from_bin):
    df = pd.concat([df_features, labels], axis=1)
    bins = pd.qcut(df['K2'], num_bins, labels=False)
    features_train_strat, label_train_strat, features_test_start, label_test_strat = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(num_bins):
        count = sum(bins == i)
        indices = list(df[bins == i].index)
        set_size = math.ceil(count * perecent_from_bin)
        a, b, c, d = random_sampling(df_features, labels, set_size, indices)
        features_train_strat = pd.concat([features_train_strat, a])
        label_train_strat = pd.concat([label_train_strat, b])
        features_test_start = pd.concat([features_test_start, c])
        label_test_strat = pd.concat([label_test_strat, d])
    return features_train_strat, label_train_strat, features_test_start, label_test_strat