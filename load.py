import torch
import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from attrib import DatasetAttributes
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from sklearn.model_selection import train_test_split
import copy

def load_train_and_val_data(file_name, le):
    '''Loads data as numpy arrays and converts them to tensors.
    Returns TensorDatasets containing inputs and outputs for validation and training.'''
    inputs, outputs, train_attrib = load_data(
        file_name, le, True)
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs.numpy(), outputs.numpy(), train_size = 0.96)
    val_attrib = copy.deepcopy(train_attrib)
    val_attrib.num_examples = val_outputs.shape[0]
    train_attrib.num_examples = train_outputs.shape[0]
    return make_tensor_dataset(train_inputs, train_outputs), make_tensor_dataset(val_inputs, val_outputs), train_attrib, val_attrib

def load_test_data(file_name, le):
    inputs, outputs, attrib = load_data(file_name, le, False)
    return TensorDataset(inputs, outputs), attrib

def make_tensor_dataset(inputs, outputs):
    tensor_inputs = torch.tensor(inputs, dtype=torch.float)
    tensor_outputs = torch.tensor(outputs, dtype=torch.long)
    return TensorDataset(tensor_inputs, tensor_outputs)

def load_data(file_name, le, train_mode):
    '''Loads data as numpy arrays and converts them to tensors.
    Returns a TensorDataset containing inputs and outputs.'''
    attrib = DatasetAttributes()
    # load the face embeddings
    data = pickle.loads(open(file_name, "rb").read())
    if train_mode:
        le.fit(data["names"])
    labels = le.transform(data["names"])
    tensor_inputs = torch.tensor(data['embeddings'], dtype=torch.float)
    tensor_outputs = torch.tensor(labels, dtype=torch.long)
    attrib.num_classes = max(attrib.num_classes, torch.max(tensor_outputs).item() + 1)
    attrib.num_examples = tensor_outputs.shape[0]
    attrib.feature_length = tensor_inputs.shape[1]
    return tensor_inputs, tensor_outputs, attrib

def build_dataloaders():
    '''Loads data from files into batched tensor dataloaders.'''

    train_embeddings_file = "output/embeddings.pickle"
    test_embeddings_file = "output/test_embeddings.pickle"
    le_file = "output/le.pickle"
    le = LabelEncoder()

    train_dataset, val_dataset, train_attrib, val_attrib = load_train_and_val_data(
        train_embeddings_file, le)
    test_dataset, test_attrib = load_test_data(
        test_embeddings_file, le)
    test_attrib.num_classes = max(train_attrib.num_classes, test_attrib.num_classes)
    train_attrib.num_classes = test_attrib.num_classes
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
    }
    attrib_dict = {
        'train': train_attrib,
        'val': val_attrib,
        'test': test_attrib,
    }
    dataloaders = {x: DataLoader(datasets[x], batch_size=1024,
                                 num_workers=4, shuffle=True,
                                 pin_memory=torch.cuda.is_available()
                                 ) for x in ['train', 'val', 'test']}
    return dataloaders, attrib_dict

if __name__ == '__main__':
    # Just preprocess, save the data, build the dataloaders, and throw them out.
    # Used to profile loading speed
    build_dataloaders()
