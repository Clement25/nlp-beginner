from utils import CONLL2003, sequence_mask, init_linear_weights
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

# class CRF(nn.Module):
#     """Implement the conditional random field described in original paper
#     """
#     def __init__(self, input_dim, num_class):
#         super(CRF, self).__init__()

#         self.input_dim = input_dim
#         self.num_class = num_class
#         self.init_linear = nn.Linear(input_dim, num_class) # Initial probability matrix
#         self.trans_linear = nn.Linear(input_dim, num_class**2)  # since we have num_class, we have num_class*num_class transition probabilities
    
#     def forward(self, x, labels, lengths):
#         """ Forward propogation of CRF
#         Params:
#             x (Tensor): Tensor of BiLSTM output hiddens with shape (batch_size, seq_len, 2*hidden)
#             labels (Tensor): Tensor of the ground truth of labels with shape (batch_size, seq_len)
#             length (Tensor): The lengths of all sequence in the batch, with shape (batch_size,)
#         Returns:
#             The returns depend on the mode (train/eval) of this model
#             In "train" mode:
#                 log_probs (Tensor): Tensor of logarithmic probabilities of batch sequences
#             In "eval" mode:
#                 predict_out (Tensor): Tensor of predicted labels for input sequence batch
#         """
#         init = torch.exp(self.init_linear(x[:,0,:]))  # (batch_size, num_class)
#         trans = torch.exp(self.trans_linear(x[:,1:,:])).view(x.size(0), x.size(1)-1, self.num_class, self.num_class)  # (batch_size, seq_len-1, num_class, num_class)
        
#         if self.training:
#             # Compute the sequence probability
#             log_probs = self._chain_forward(init, trans, labels, lengths)
#             return log_probs.sum()
#         else:
#             predict_out = self._decode(init, trans, labels, lengths)
#             return predict_out

#     def _chain_forward(self, init, trans, labels, lengths):
#         probs = torch.zeros(size=(init.size(0),))

#         # batch iteration
#         for i, length in enumerate(lengths):
#             init_tag = labels[i,0]
#             scores = init[i,:]  # (num_class,)
#             path_score = init[i,init_tag]   # A number

#             # sequence iteration
#             for j in range(1,length):
#                 start, end = labels[i,j-1], labels[i,j]
#                 path_score = path_score * trans[i, j-1, end, start]
#                 scores = trans[i,j-1] * scores

#             probs[i] = path_score/(torch.sum(scores).item())
        
#         return torch.log(probs)
    
#     def _decode(self, init, trans, labels, lengths):
#         probs = torch.zeros(size=(init.size(0),))

#         # -1 is a meaningless label
#         batch_label_seq = torch.full((labels.size()),fill_value=-1)
#         for i, length in enumerate(lengths):
#             init_tag = labels[i,0]
#             scores = init[i,:]  # (num_class,)

#             if length > 1:
#                 back_pointers = torch.IntTensor(length-1, self.num_class)

#                 # Use viterbi algorithm to decode input sequence
#                 for j in range(1,length):
#                     transition = trans[i,j-1]
#                     trans_prob = transition.transpose(0,1)*scores
#                     maxv, maxi = torch.topk(trans_prob,1,dim=1)  # (num_class,) , (num_class,)

#                     back_pointers[j-1] = maxi.squeeze(-1)
#                     scores = maxv
                
#                 max_end = torch.argmax(scores)

#                 label_seq = [max_end]
#                 for j in range(1,length):
#                     cur_node = label_seq[-1]
#                     label_seq.append(back_pointers[-j,cur_node].item())
#                 label_seq.reverse()
#             else:
#                 label_seq = [torch.argmax(scores).item()]

#             # pdb.set_trace()
#             batch_label_seq[i][:length] = torch.Tensor(label_seq)+2  # Note first two labels in label vocab are '<unk>' and '<pad>' respectively
        
#         return batch_label_seq


class BLCC(nn.Module):
    """Implement the NER network proposed in the paper: <https://arxiv.org/pdf/1603.01354.pdf>
    """
    def __init__(self, num_word_embeddings, num_char_embeddings, num_class, 
                    embedding_dim_word=100, embedding_dim_char=30,
                    window_size=3, bidirectional=True,
                    hidden_size=100, p=0.5, word_vectors=None, device='cuda'):
        """ 
        Arguments:
            num_word_embeddings (int): The number of tokens making up the emebedding table
            embedding_dim_word (int): The dimension of the word vector of each token
            num_char_embeddings (int): The 
        """
        super(BLCC, self).__init__()

        if word_vectors is None:
            self.word_embedding_layer = nn.Embedding(num_word_embeddings, embedding_dim_word)
        else:
            self.word_embedding_layer = nn.Embedding.from_pretrained(word_vectors)

        # self.char_embedding_layer = nn.Embedding(num_char_embeddings, embedding_dim_char)
        self.char_embedding_layer = nn.Embedding.from_pretrained(self._init_char_embedding(num_char_embeddings, embedding_dim_char))
        self.char_embedding_layer.freeze = False
        self.num_class = num_class
        self.dropout = nn.Dropout(p)
        self.charCNN = nn.Conv1d(
                                    in_channels=embedding_dim_char, 
                                    out_channels=embedding_dim_char, 
                                    kernel_size=window_size,
                                    padding=window_size-1
                                )
        self.BiLSTM = nn.LSTM(  
                                embedding_dim_word, \
                                # + embedding_dim_char, 
                                hidden_size, 
                                num_layers=1, 
                                bias=False,
                                batch_first=True, 
                                dropout=p, 
                                bidirectional=bidirectional
                            )

        self.crf = CRF(self.num_class, True)
        self.crf.apply(init_linear_weights)
        self.hidden2tag = nn.Linear((2 if bidirectional else 1)*hidden_size, num_class)
        self.device = device

    def forward(self, input_words, input_char, input_labels):
        """ Forward propogation
        Params:
            input_words (Tensor): Tensor of indices of words in batch sequences with shape (batch_size, max_seqlen, word_embedding_dim)
            input_char (Tensor): Tensor of indices of characters in batch sequence with shape (batch_size, max_seqlen, max_wordlen, char_emebedding_dim)
            input_labels (Tensor): Tensor of indices of labels in batch sequence with shape (batch_size, num_class)
        Returns:
            crf_out (Tensor): Tensor of CRF outputs. The meaning depends on model's mode. In traning mode the outputs are log likelihood.
                            In eval mode the outputs are decoding results (batch of sequence labels).
        """
        if not self.training:
            assert (not (input_labels is None))

        input_word_raw, input_seqlen = input_words

        word_repr = self.word_embedding_layer(input_word_raw)    # (batch_size, max_seqlen, embedding_dim_word)

        char_repr = self._forward_char(input_char)   # (batch_size, max_seqlen, num_filters)
        word_embeds = torch.cat([char_repr, word_repr], dim=-1) # (batch_size, max_seqlen, num_fiters+embedding_dim_word)

        word_embeds = self.dropout(word_embeds)
        # word_embeds = self.dropout(word_repr)
        packed_sequence = pack_padded_sequence(word_embeds, input_seqlen, batch_first=True, enforce_sorted=False)
        out, _ = self.BiLSTM(packed_sequence)  # (batch_size, max_seqlen, 2*hidden_size)
        lstm_out = pad_packed_sequence(out, batch_first=True)[0]   # (batch_size, seqlen, 2*hidden_size)
        lstm_out = self.dropout(lstm_out)

        crf_mask = self._get_crf_mask(input_seqlen)
        emission = self.hidden2tag(lstm_out)

        if self.training:
            crf_out = self.crf(emission, input_labels - 2, crf_mask) # skip <unk> and <pad> token
        else:
            crf_out = self.crf.decode(emission, crf_mask)
        return crf_out

    def _get_crf_mask(self, seqlen):
        max_seqlen = seqlen.max().item()
        csum = torch.ones(size=(seqlen.size(0),max_seqlen),device=self.device).cumsum(dim=1)
        return csum <= seqlen.unsqueeze(-1)

    def _forward_char(self, input_char):
        """ Forward propogation of the convolutional neural network on character level. 
        Params:
            input_char (Tuple): A triplet created by torchtext.data.nexted_field, each element represents: character indices, lengths of sequence, lengths of each word.
        Returns:
            char_repr (Tensor): The result char embeddings processed by charcnn network. The embedding size equals to the number of filters.
        """
        char_inputs, word_lens, char_lens = input_char
        batch_size, max_seqlen = char_lens.size(0), char_lens.size(1)

        # assert char_inputs.size(-2) == word_lens.max().item() and char_inputs.size(-1) == char_lens.max().item()
        
        char_inputs_flatten = char_inputs.view(-1, char_inputs.size(-1))  # (batch_size*max_seqlen, max_wordlen)
        char_embeds = self.char_embedding_layer(char_inputs_flatten).permute(0,2,1)  # (batch_size*max_seqlen, embedding_dim_char, max_wordlen)
        char_embeds = self.dropout(char_embeds)

        char_conv = self.charCNN(char_embeds)  # (batch_size*max_seqlen, num_filters, max_wordlen+2)
        char_conv = char_conv.view(batch_size, max_seqlen, char_conv.size(1), char_conv.size(2))    # (batch_size, max_seqlen, num_filters, max_wordlen+2)

        char_conv = char_conv.permute(0,1,3,2) # (batch_size, max_seqlen, max_wordlen+2, num_filters)
        char_mask = sequence_mask(char_conv, (char_lens > 0) * (char_lens+2), self.device)  # if it's a padding word in some sequence, just mask it
        char_conv = char_conv.masked_fill(char_mask, -10000.0)

        char_conv = char_conv.view(char_conv.size(0)*char_conv.size(1), char_conv.size(2), char_conv.size(3)).permute(0,2,1)    # (batch_size*max_seqlen, num_filters, max_wordlen+2)
        char_repr = F.max_pool1d(char_conv, kernel_size=char_conv.size(-1)).squeeze(-1) # (batch_size*max_seqlen, num_filters)
        char_repr = char_repr.view(batch_size, max_seqlen, -1) # (batch_size, max_seqlen, num_filters)
        return char_repr

    def _init_char_embedding(self, num_emebedding, embedding_dim):
        res = torch.empty(size=(num_emebedding, embedding_dim))
        bound = torch.sqrt(torch.Tensor([3/embedding_dim]))
        return nn.init.uniform_(res, -bound.item(), bound.item())