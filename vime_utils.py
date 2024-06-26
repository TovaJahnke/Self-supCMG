import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, train_test_split
from random import choice
from torch import nn



class preloader(Dataset):
    def __init__(self, dataset, normalize=True):
        self.dataset = dataset
        self.df=pd.read_csv(dataset, sep='\t')
        self.df_labels=self.df[["N0"]]
        self.df=self.df.drop(columns=["ID", "N0"])
        # normalize all values (z-score)
        if normalize == True:
            for column in self.df.columns:
                self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        self.dataset=torch.tensor(self.df.to_numpy()).float()
        self.labels=torch.tensor(self.df_labels.to_numpy().reshape(-1)).long()
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]
    def get_column(self, column_name):
        return self.df[column_name]
    
   
class preloader_synt(Dataset):
    def __init__(self, dataset,num_rows=10000, normalize=True):
        self.dataset = dataset
        self.df=pd.read_csv(dataset, sep='\t', nrows=num_rows)
        #self.df_labels=self.df[["N0"]]
        self.df=self.df.drop(columns=["ID"])
        # normalize all values (z-score)
        if normalize == True:
            for column in self.df.columns:
                self.df[column] = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        self.dataset=torch.tensor(self.df.to_numpy()).float()
       # self.labels=torch.tensor(self.df_labels.to_numpy().reshape(-1)).long()
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    """
    def __getitem__(self, idx):
        x = self.dataset[idx]
        x_shape = x.shape
        return x, x_shape, self.labels[idx]
"""

    
    #USED TO FIND BINARY CORRELATIONS 
    
class preloaderspecial(Dataset):
    def __init__(self, dataset, normalize=True):
        self.dataset = dataset
        self.df=pd.read_csv(dataset, sep='\t')
        self.df_labels=self.df[["N0"]]
        self.df=self.df.drop(columns=["ID", "N0"])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]
    def get_column(self, column_name):
        return self.df[column_name]
        
    
