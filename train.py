import torch
from torch import autograd, nn
import torch.nn.functional as F

from torchtext import data, datasets
from torchtext.vocab import GloVe

import numpy as np
from rnn import RNN

def train(data_path, train_path, val_path, test_path, hidden_size,
        num_classes, num_layers, num_dir, batch_size, emb_dim, dropout,
        net_type, embfix):

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
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=emb_dim), min_freq=2)
    prevecs=TEXT.vocab.vectors
    #TEXT.build_vocab(train, min_freq=3)
    LABEL.build_vocab(train)

    # build iterators
    train_iter = data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.text), train=True)
    val_iter = data.Iterator(val, batch_size=batch_size, repeat=False, train=False, sort=False, shuffle=False)
    test_iter = data.Iterator(test, batch_size=len(test), repeat=False, train=False, sort=False, shuffle=False)

    # print info
    print(max(LABEL.vocab.freqs.values()))
    print('num_classes: ',len(LABEL.vocab))
    print('input_size: ', len(TEXT.vocab))

    print('majority class acc:', max(LABEL.vocab.freqs.values()) / len(train))
    print('random guess acc:',
            (max(LABEL.vocab.freqs.values()) / len(train)) ** 2
            + (min(LABEL.vocab.freqs.values()) / len(train)) ** 2)

    num_classes = len(LABEL.vocab)
    input_size = len(TEXT.vocab)

    model = RNN(input_size=input_size,
                    hidden_size=hidden_size,
                    num_classes=num_classes,
                    prevecs=prevecs,
                    num_layers=num_layers,
                    num_dir=num_dir,
                    batch_size=batch_size,
                    emb_dim=emb_dim,
                    embfix=embfix,
                    dropout=dropout,
                    net_type=net_type)

    epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.5)
    if int(torch.cuda.is_available()) == 1:
        model = model.cuda()

    # train
    model.train()
    best_val_acc = 0
    for e in range(epochs):
        print('Epoch:', e)
        tot_loss = 0
        corrects = 0
        train_iter.repeat=False
        for batch_count,batch in enumerate(train_iter):
            #print('Batch:', batch_count)
            #print(batch.text)
            #print(batch.label)
            model.zero_grad()

            inp = batch.text.t()
            preds = model(inp)
            target = batch.label

            #print(preds, batch.label)
            loss = criterion(preds, batch.label)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(preds, 1)
            corrects += int(preds.data.eq(target.data).sum())
            tot_loss += loss.data[0]

        print('acc (train):', 100 * corrects / len(train_iter.dataset))
        print('loss (train):', tot_loss)
        val_acc, _, val_loss = evaluate(val_iter, model, TEXT, LABEL)
        print('acc (val):', val_acc)
        print('loss (val):', val_loss)
        if val_acc > best_val_acc:
            test_acc, test_preds, test_loss = evaluate(test_iter, model, TEXT, LABEL)
            #print('Test acc:', test_acc)
            f = open('./preds/preds_' + str(e) + '.txt', 'w')
            for x in test_preds:
                f.write(str(int(x)) + '\n')
            f.close()
            torch.save(model.state_dict(), './models/e' + str(e) + '_' + str(val_acc) + '.pt')

def evaluate(data_iter, model, TEXT, LABEL):
    model.eval()
    tot_loss = 0
    corrects = 0
    all_preds = np.array([]) # preds for text file

    for batch_count,batch in enumerate(data_iter):

        inp = batch.text.t()
        preds = model(inp)
        target = batch.label

        loss = F.cross_entropy(preds, batch.label)
        tot_loss += loss.data[0]


        _, preds = torch.max(preds, 1)

        all_preds = np.append(all_preds, preds.data)
        #print('preds:', preds.data)
        #print('targets:', target.data)
        #print('sum:', int(preds.data.eq(target.data).sum()))
        corrects += int(preds.data.eq(target.data).sum())
    val_acc = 100 * corrects / len(data_iter.dataset)

    #val_preds =
    return val_acc, all_preds, tot_loss

def main():
    data_path = './data/'
    train_path = 'train.tsv'
    val_path = 'val.tsv'
    test_path = 'test.tsv'

    # hyperparams
    hidden_size = 128
    num_classes = 2
    num_layers = 2
    num_dir = 2
    batch_size = 8
    emb_dim = 300
    dropout = .2
    net_type = 'lstm'
    embfix=False

    train(data_path=data_path,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            hidden_size=hidden_size,
            num_classes = num_classes,
            num_layers = num_layers,
            num_dir = num_dir,
            batch_size = batch_size,
            emb_dim = emb_dim,
            dropout = dropout,
            net_type = net_type,
            embfix = embfix)

if __name__ == '__main__':
    main()
