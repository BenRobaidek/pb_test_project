import torch
from torch import autograd, nn
import torch.nn.functional as F
from torch.autograd import Variable

from rnn import RNN
from torchtext import data, datasets
from torchtext.vocab import GloVe
from train import evaluate

def main():
    cuda = int(torch.cuda.is_available())-1

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

    model = RNN().load_state_dict(torch.load('mytraining.pt'))

if __name__ == '__main__':
    main()
