import pandas as pd
import numpy as np
import re
import random

def main():

    # read data
    print('Preprocessing...')
    data = pd.read_csv('./data/testProject_About.labeled.csv', header=None)
    train_test_split = pd.read_csv('./data/About.train_test_split.csv', header=0)

    # preprocess
    for i, row in enumerate(data[0]):
        row = row.lower()
        row = row.replace('...', ' ... ')
        row = row.replace(':', ' : ')
        row = row.replace(';', ' ; ')
        row = row.replace('$', ' $ ')
        row = row.replace('£', ' £ ')
        row = row.replace('%', ' % ')
        row = row.replace('(', ' ( ')
        row = row.replace(')', ' ) ')
        row = row.split()
        for j, word in enumerate(row):
            row[j] = re.sub(r'([,.!?]+$)', r' \1 ',word)
            # handle numbers with units
            row[j] = re.sub(r'(\d+)([A-z]{1,2})', r'\1 \2', row[j])
        row = ' '.join(row)
        data[0][i] = row
        #print(row)

    # split train/val/test
    train_val = np.array(data[train_test_split['set'] == 'train'][[0,1]])
    np.random.seed(0)
    np.random.shuffle(train_val)
    test = np.array(data[train_test_split['set'] == 'test'][[0,1]])

    train = train_val[:-1000]
    val = train_val[-1000:]
    #test = np.array(test[[0,1]])

    # random under-sampling
    """
    for index, row in train.iterrows():
        if row[1] == 'None' and random.random() < .5:
            train = train.drop(index)
    """

    # write to file
    write2file(train, './data/train.tsv')
    write2file(val, './data/val.tsv')
    write2file(test, './data/test.tsv')
    print('.tsv files saved')

    print(len(train), ' examples in train')
    print(len(val), ' examples in val')
    print(len(test), ' examples in test')

def write2file(arr, file_path):
    f = open(file_path, 'w')
    for x in arr:
        f.write(x[0].strip() + '\t' + x[1].strip() + '\n')
    f.close()

if __name__ == '__main__':
    main()
