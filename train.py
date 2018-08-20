import torch
from torch import autograd, nn
import torch.nn.functional as F

from torchtext import data, datasets
from torchtext.vocab import GloVe

import numpy as np
from rnn import RNN

def train(data_path, train_path, val_path, test_path, bs):
    print('Training...')

    # define fields
    TEXT = data.Field(lower=True,init_token="<start>",eos_token="<end>")
    LABEL = data.Field(sequential=False, unk_token=None)

    # build dataset splits
    train, val, test = data.TabularDataset.splits(
        path=data_path, train=train_path,
        validation=val_path, test=test_path, format='tsv',
        fields=[('text', TEXT), ('label', LABEL)])

    # build vocabs
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100), min_freq=2)
    #TEXT.build_vocab(train, min_freq=3)
    LABEL.build_vocab(train)

    # build iterators
    train_iter = data.BucketIterator(train, batch_size=bs, sort_key=lambda x: len(x.text), train=True)
    val_iter = data.Iterator(val, batch_size=bs, repeat=False, train=False, sort=False, shuffle=False)
    test_iter = data.Iterator(test, batch_size=len(test), repeat=False, train=False, sort=False, shuffle=False)

    # print info
    print(max(LABEL.vocab.freqs.values()))
    print('num_classes: ',len(LABEL.vocab))
    print('input_size: ', len(TEXT.vocab))

    print('majority class acc:', max(LABEL.vocab.freqs.values()) / len(train))
    print('random guess acc:',
            (max(LABEL.vocab.freqs.values()) / len(train)) ** 2
            + (min(LABEL.vocab.freqs.values()) / len(train)) ** 2)


    input_dim = len(TEXT.vocab)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 2

    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)

    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters())
    if int(torch.cuda.is_available()) == 1:
        model = model.cuda()

    # train
    model.train()
    for e in range(epochs):
        print('Epoch:', e)
        tot_loss = 0
        train_iter.repeat=False
        for batch_count,batch in enumerate(train_iter):
            #print('Batch:', batch_count)
            print(batch.text)
            print(batch.label)
            model.zero_grad()
            inp = batch.text
            preds = model(inp)
            loss = criterion(preds, batch.label)
            loss.backward()
            optimizer.step()
            tot_loss += loss.data[0]
        print('Loss:,', tot_loss)
        val_acc = evaluate(val_iter, model, TEXT, LABEL)
        print('Validation Acc:', val_acc)

def evaluate(data_iter, model, TEXT, LABEL):
    model.eval()
    corrects = 0
    for batch_count,batch in enumerate(data_iter):
        inp = batch.text
        preds = model(inp)
        target = batch.label

        #loss = F.cross_entropy(preds, batch.label)

        _, preds = torch.max(preds, 1)
        #print('preds:', preds.data)
        #print('targets:', target.data)
        #print('sum:', int(preds.data.eq(target.data).sum()))
        corrects += int(preds.data.eq(target.data).sum())
    return 100 * corrects / len(data_iter.dataset)

def main():
    data_path = './data/'
    train_path = 'train.tsv'
    val_path = 'val.tsv'
    test_path = 'test.tsv'
    train(data_path=data_path, train_path=train_path, val_path=val_path,
            test_path=test_path, bs=8)

if __name__ == '__main__':
    main()
