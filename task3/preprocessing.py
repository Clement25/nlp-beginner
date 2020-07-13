from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch

class SNLIDataset(object):
    def __init__(self, embedding_dim, batch_size, device='cuda'):
        # set up fields
        self.TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
        self.LABEL = data.Field(sequential=False)

        #make splits for data
        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

        #build the vocabulary
        self.TEXT.build_vocab(self.train, vectors=GloVe(name='6B', dim=embedding_dim))

        self.LABEL.build_vocab(self.train)

        # make iterator for splits
        self.train_iter, self.dev_iter, self.test_iter =    \
            data.BucketIterator.splits((self.train, self.dev, self.test), batch_size=batch_size, device=device)

    @property
    def vocab_size(self):
        return len(self.TEXT.vocab)
    
    @property
    def num_class(self):
        return len(self.LABEL.vocab)
    
    def _data_size(self, name='train'):
        try:
            return {'train': len(self.train), 
                    'dev':len(self.dev), 
                    'test':len(self.test)}[name]
        except:
            raise ValueError("Dataset name must be one of 'train', 'dev' and 'test'\n")
    
    @property
    def train_size(self):
        return self._data_size('train')
    
    @property
    def dev_size(self):
        return self._data_size('dev')

    @property
    def test_size(self):
        return self._data_size('test')
    
    @property
    def pretrained_vectors(self):
        return self.TEXT.vocab.vectors

if __name__ == '__main__':
    ds = SNLIDataset(50, 128)
    itos = ds.TEXT.vocab.itos
    print(type(itos))
    print(len(itos))
    print(ds.TEXT.vocab.stoi['zuma'])