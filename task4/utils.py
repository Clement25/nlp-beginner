from torchtext import data
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import GloVe, CharNGram
import torch
import torch.nn as nn
import pdb

PAD_TOKEN="<pad>"

class CONLL2003(object):
    def __init__(self, batch_size=128, word_embedding_dim=100, tag_type='pos', path='conll2003_polished', \
                    sep_name={'train':'train.txt', 'valid':'valid.txt','test':'test.txt'},  \
                    device='cuda'):
        """
        conll 2003 dataset:
            Extract Conll2003 dataset using torchtext.
            Applies Glove.6B.200d and char N-gram pretrained vectors.
            Set up word and character field. 
        Params:
            @batch_size (int): params dataset to initialize training iterator only
            @embedding_dim (int): input word and character, we use Glove so the dim is the Glove dim
            @tag(str): the tag used as labels (1 of 3)
            @path(str): root path to store the entire dataset
            @sep_name(dic[sep_name]->path): map separation name to corresponding file
            @device(str): specify the device of the dataset
        """
        # set up fields
        self.inputs_word = data.Field(lower=True, batch_first=True, include_lengths=True)

        self.inputs_char_nesting = data.Field(tokenize=list, pad_token="<pad>", batch_first=True, lower=True)
        self.inputs_char = data.NestedField(self.inputs_char_nesting, pad_token="<pad>",include_lengths=True)

        self.labels = data.Field(batch_first=True)

        #build the vocabulary
        fields = [(('inputs_word','inputs_char'),(self.inputs_word, self.inputs_char))] +  \
                    [('labels', self.labels) if tag==tag_type else (None,None) for tag in ['pos','chunk', 'ner']]
        
        self.train, self.val, self.test = SequenceTaggingDataset.splits(
            path=path,
            train=sep_name['train'],
            validation=sep_name['valid'],
            test=sep_name['test'],
            separator=' ',
            fields=fields
        )

        self.inputs_char.build_vocab(self.train.inputs_char, self.val.inputs_char, self.test.inputs_char)
        self.inputs_word.build_vocab(self.train.inputs_word, self.val.inputs_word, self.test.inputs_word, max_size=50000, \
            vectors=GloVe(name='6B', dim=word_embedding_dim))
        self.labels.build_vocab(self.train.labels)

        # make iterator for splits
        self.train_iter, self.val_iter, self.test_iter =    \
            data.BucketIterator.splits((self.train, self.val, self.test), 
                                    batch_sizes=(batch_size, batch_size, batch_size), shuffle=True, device=device)
    
    @property
    def char_vocab_size(self):
        return len(self.inputs_char.vocab)

    @property
    def word_vocab_size(self):
        return len(self.inputs_word.vocab)
    
    @property
    def num_class(self):
        return len(self.labels.vocab)
    
    def _separation_size(self, name='train'):
        try:
            return {'train': len(self.train), 
                    'val':len(self.val), 
                    'test':len(self.test)}[name]
        except:
            raise ValueError("Separation name must be one of 'train', 'val' and 'test'\n")
    
    @property
    def train_size(self):
        return self._separation_size('train')
    
    @property
    def val_size(self):
        return self._separation_size('val')

    @property
    def test_size(self):
        return self._separation_size('test')
    
    def _padding_idx(self, name='word'):
        return {'word':self.inputs_word.vocab.stoi[PAD_TOKEN],
                'char':self.inputs_char.vocab.stoi[PAD_TOKEN]}[name]
    
    @property
    def word_padding_idx(self):
        return self._padding_idx(name='word')

    @property
    def char_padding_idx(self):
        return self._padding_idx(name='char')

    def _pretrained_vectors(self, name='word'):
        return {'word':self.inputs_word.vocab.vectors,
                'char':self.inputs_char.vocab.vectors}[name]

    @property
    def word_vectors(self):
        return self._pretrained_vectors(name='word')
    
    @property
    def char_vectors(self):
        return self._pretrained_vectors(name='char')

def sequence_mask(input_tensor, lengths, device, maxlen=None, dtype=torch.bool):
    """Create a mask for the input_tensor and a given list of lengths.
    Params:
        input_tensor (Tensor): A tensor with shape (..., max_seq_len, embedding_size), the most likely shape of tensors for language model
        lengths (List of int or Tensor): A tensor contains the lengths of all sequences in the input_tensor
    """
    if maxlen is None:
        maxlen = lengths.max()
    # mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()

    # the last dimension is embedding so we don't care about it
    mask = torch.ones(input_tensor.size()[:-1], device=device)    # (batch_size, max_seq_len)
    mask = ~(mask.cumsum(dim=-1)>lengths.unsqueeze(-1))
    mask = mask.type(dtype).unsqueeze(-1)
    return mask

def init_linear_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.0)


if __name__ == '__main__':
    ds = CONLL2003()
    # 看一下token对应的embedding
    # char_iter = ds.val_iter
    train_iter = ds.train_iter
    val_iter = ds.val_iter
    for data1, data2 in zip(train_iter,val_iter):
        pdb.set_trace()