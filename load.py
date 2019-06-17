import torch
import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from attrib import DatasetAttributes
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def load_data(file_name, le, train_mode):
    '''Loads data as numpy arrays and converts them to tensors.
    Returns a TensorDataset containing inputs and outputs.'''
    attrib = DatasetAttributes()
    # load the face embeddings
    #print("[INFO] loading face embeddings...")
    data = pickle.loads(open(file_name, "rb").read())
    if train_mode:
        le.fit(data["names"])
    labels = le.transform(data["names"])
    tensor_inputs = torch.tensor(data['embeddings'], dtype=torch.float)
    #print(tensor_inputs)
    tensor_outputs = torch.tensor(labels, dtype=torch.long)
    #print(tensor_outputs)
    attrib.num_classes = max(attrib.num_classes, torch.max(tensor_outputs).item() + 1)
    #print("num classes " + str(attrib.num_classes))
    attrib.num_examples = tensor_outputs.shape[0]
    #print(attrib.num_examples)
    attrib.feature_length = tensor_inputs.shape[1]
    #print(attrib.feature_length)
    return TensorDataset(tensor_inputs, tensor_outputs), attrib

def build_dataloaders():
    '''Loads data from files into batched tensor dataloaders.'''

    train_embeddings_file = "output/embeddings.pickle"
    val_embeddings_file = "output/val_embeddings.pickle"
    le_file = "output/le.pickle"
    le = LabelEncoder()

    train_dataset, train_attrib = load_data(train_embeddings_file, le, True)
    print("train1 " + str(train_attrib.num_examples))
    val_dataset, val_attrib = load_data(val_embeddings_file, le, False)
    val_attrib.num_classes = max(train_attrib.num_classes, val_attrib.num_classes)
    train_attrib.num_classes = val_attrib.num_classes
    print("train " + str(train_attrib.num_examples))
    datasets = {
        'train': train_dataset,
        'val': val_dataset
    }
    attrib_dict = {
        'train': train_attrib,
        'val': val_attrib
    }
    dataloaders = {x: DataLoader(datasets[x], batch_size=1024,
                                 num_workers=4, shuffle=True,
                                 pin_memory=torch.cuda.is_available()
                                 ) for x in ['train', 'val']}
    return dataloaders, attrib_dict

if __name__ == '__main__':
    # Just preprocess, save the data, build the dataloaders, and throw them out.
    # Used to profile loading speed
    build_dataloaders()
