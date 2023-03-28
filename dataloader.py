import torch
import numpy as np
import torchvision.models
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

class MandatoryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.label_encode = {'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = []
        self.labels = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                self.filenames.append((os.path.join(subdir, file)))
                self.labels.append(self.label_encode[os.path.basename(subdir)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #TODO
        filename = self.filenames[idx]
        image = Image.open(filename).convert('RGB')
        label = torch.tensor(self.labels[idx])
        if self.transform != None:
            image = self.transform(image)

        sample = (image, label, filename)
        return sample
    
    def targets(self):
        return self.labels
            
class Subset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = [dataset.labels[idx] for idx in indices]
        self.filenames = [dataset.filenames[idx] for idx in indices]
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return len(self.indices)
    
# This only loads the images, since the labels are not needed in this case
class ImageNet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(root_dir))
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.filenames[idx])
        img = Image.open(filename).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return (img, 0, filename)

def load_splits(dataset, batch_size, num_workers=0, ratio=[0.117, 0.176], seed=-1):
    # ratio -> [val_set, test_set]
    labels = dataset.targets()
    indices = list(range(len(dataset)))

    train_indices, test_indices, train_labels, test_labels = train_test_split(indices, labels, test_size=ratio[1], stratify=labels, random_state=seed)
    train_indices, val_indices, train_labels, val_labels = train_test_split(train_indices, train_labels, test_size=ratio[0]/(1-ratio[1]), stratify=train_labels, random_state=seed)
    
    assert (len(train_indices) + len(test_indices) + len(val_indices)) == len(dataset)
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader# Plots a histogram of the class distribution for a given dataset

def load_CIFAR(PATH, batch_size, transform, download=True, num_workers=0):
    dataset = torchvision.datasets.CIFAR100(root=PATH, train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader

def load_ImageNet(PATH, batch_size, transform, num_workers=0):
    imagenet_dataset = ImageNet(root_dir=PATH, transform=transform)
    dataloader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def check_disjoint(train, val, test):
    filenames = train.dataset.filenames + val.dataset.filenames + test.dataset.filenames
    assert len(filenames) == len(set(filenames))
    print("Datasets are disjoint.")

def plot_class_dist(dataset):
    targets = dataset.targets

    class_counts = pd.Series(targets).value_counts()

    plt.bar(class_counts.index, class_counts.values)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def class_dist_similarity(original, train, val, test):
    # Conducting a chi-squared test to verify that all class-distributions are the same
    og_targets = original.targets()
    tr_targets = train.targets
    va_targets = val.targets
    te_targets = test.targets
    
    og_counts = pd.Series(og_targets).value_counts()
    tr_counts = pd.Series(tr_targets).value_counts()
    va_counts = pd.Series(va_targets).value_counts()
    te_counts = pd.Series(te_targets).value_counts()
    
    counts_df = pd.concat([og_counts, tr_counts, va_counts, te_counts], axis=1, sort=True)
    counts_df.columns = ['Original', 'Train', 'Validation', 'Test']
    counts_df.fillna(0, inplace=True)    
     
    _, pval, _, _ = chi2_contingency(counts_df) # returns the P-value, should be close to 1.0
    
    print('P-value from chi-squared test: %f'%(pval))
