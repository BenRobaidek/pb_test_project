import pandas as pd
import numpy as np
import re

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
        row = row.split()
        for j, word in enumerate(row):
            row[j] = re.sub(r'([,.!?]+$)', r' \1 ',word)
            # handle numbers with units
            row[j] = re.sub(r'(\d+)([A-z]{1,2})', r'\1 \2', row[j])
        row = ' '.join(row)
        data[0][i] = row
        #print(row)

    # split train/test
    train = data[train_test_split['set'] == 'train']
    test = data[train_test_split['set'] == 'test']

    # write to file
    train[[0,1]][:-1000].to_csv('./data/train.csv', sep='\t', header=False, index=False)
    train[[0,1]][-1000:].to_csv('./data/val.csv', sep='\t', header=False, index=False)
    test[[0,1]].to_csv('./data/test.csv', sep='\t', header=False, index=False)
    print('.csv files saved')

    print(len(train) - 1000, ' examples in test')
    print('1000 examples in val')
    print(len(test), ' examples in test')

if __name__ == '__main__':
    main()
