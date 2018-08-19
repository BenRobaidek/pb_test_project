import torch
from torch import autograd, nn

from torchtext import data, datasets

import numpy as np

def train(data_path, train_path, val_path, test_path):
    print('Training...')

    # define fields
    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABEL = data.Field(sequential=False)

    # build dataset splits
    train, val, test = data.TabularDataset.splits(
        path=data_path, train=train_path,
        validation=val_path, test=test_path, format='tsv',
        fields=[('text', TEXT), ('label', LABEL)])

    # build vocabs
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)


def main():
    data_path = './data/'
    train_path = 'train.tsv'
    val_path = 'val.tsv'
    test_path = 'test.tsv'
    train(data_path=data_path, train_path=train_path, val_path=val_path,
            test_path=test_path)

if __name__ == '__main__':
    main()
