import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class InfCompositionLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, p=0.5):
        super(InfCompositionLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p)
    
    def forward(self, x, x_lens):
        """Compute Incomposition output
        Args:
            x (Tensor): shape of (batch_size, seq_len, 8*hidden_size) 
            x_lens （Tensor): 1D tensor with shape (seq_len)
        """
        x = self.linear(x)      
        x = self.dropout(x)     # (batch_size, max_seqlen, output_size)
        packed_x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed_x)   
        padded_out = (pad_packed_sequence(out))[0] # (seq_len, batch_size, hidden_size*2)

        # average pooling
        avg_out = (padded_out.sum(0))/(x_lens.reshape(-1,1).float())

        # max pooling
        max_out = (padded_out.max(0))[0]    # (batch_size, hidden_size*2)
        return avg_out, max_out

class MLP(nn.Module):
    """2-layer perceptron for classification
    """
    def __init__(self, input_size, output_size, num_class, activation='tanh', p=0.5):
        super(MLP, self).__init__()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unknown activation function!!!")
        self.dropout = nn.Dropout(p)
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, num_class)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation(x)
        out = self.linear2(x)
        return out

class NLINetwork(nn.Module):
    """Implement the model of the paper <https://arxiv.org/pdf/1609.06038.pdf> without Tree LSTM part
    """
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_class, \
            dropout=0.5, batch_size=32, bidirectional=True, vectors=None, device='cuda'):
        """
        Arguments:
             num_embeddings (int): The number of tokens that make up the embedding table
             embedding_dim (int): The dimension of embedding vectors for each token
             hidden_size (int): The vector length of LSTM hidden.
             num_layers (int): The number of layers of LSTM
             num_class (int): The number of class
        """
        super(NLINetwork, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim=embedding_dim
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.device = device
        if vectors is None:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(vectors)
        
        self.biLSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, \
                        batch_first=False, bidirectional=bidirectional, dropout=dropout)
        self.prem_hidden, self.hypo_hidden = self._init_hidden(), self._init_hidden()
        self.CompLayer = InfCompositionLayer(8*hidden_size, hidden_size, hidden_size)
        self.mlp = MLP(8*hidden_size, hidden_size, num_class)
    
    def forward(self, raw_prem, raw_hypo):
        prem, premlen = raw_prem    # (batch_size, max_seqlen)
        hypo, hypolen = raw_hypo
        
        prem_embed, hypo_embed = self.embedding(prem), self.embedding(hypo) # (batch_size, max_seqlen, embedding_size)
        prem_encoded, hypo_encoded = self.encode(prem_embed, premlen, hypo_embed, hypolen)  # (batch_size, max_seqlen, 2*hidden_size)

        assert prem_encoded.size()==(self.batch_size, premlen.max().item(), 2*self.hidden_size)
        assert hypo_encoded.size()==(self.batch_size, hypolen.max().item(), 2*self.hidden_size)

        # mask for attention
        prem_att, hypo_att = self.attention(prem, hypo, prem_encoded, hypo_encoded) # (batch_size, max_seqlen, 2*hidden_size)

        m_prem = torch.cat([prem_encoded, prem_att, prem_encoded-prem_att, torch.mul(prem_encoded, prem_att)], dim=2)   # (batch_size, max_seqlen_p, 8 * hidden_size)
        m_hypo = torch.cat([hypo_encoded, hypo_att, hypo_encoded-hypo_att, torch.mul(hypo_encoded, hypo_att)], dim=2)   # (batch_size, max_seqlen_h, 8 * hidden_size)

        prem_avg, prem_max = self.CompLayer(m_prem, premlen)
        hypo_avg, hypo_max = self.CompLayer(m_hypo, hypolen)    # (batch_size, hidden_size*2)

        mlp_input = torch.cat([prem_avg, prem_max, hypo_avg, hypo_max], dim=-1) 
        logits = self.mlp(mlp_input)
        probs = F.softmax(logits, dim=1)
        return logits, probs
        
    def attention(self, prem, hypo, prem_encoded, hypo_encoded):
        """Compute attention outputs or the alignment of 2 sentences
        Args:
            prem, hypo (Tensor): Tensor of shape (batch_size, max_seq_len), consisting of each word index
            prem_encoded, hypo_encoded (Tensor):
                Tensor of shape (batch_size, max_seq_len, 2*hiddensize), which is the output of LSTM encoder.
        """

        mask, prem_ws_mask, hypo_ws_mask = self.get_mask(prem, hypo, prem_encoded, hypo_encoded)       # (batch_size, max_seqlen_p, max_sqlen_h)
        
        hypo_encoded = hypo_encoded.transpose(1,2)  # (batch_size, 2*hidden_size, max_seqlen_h)
        
        align = torch.bmm(prem_encoded, hypo_encoded) # (batch_size, max_seqlen_p, max_seqlen_h)
        align.masked_fill_(mask < 1e-8, -1e-7)
        # pdb.set_trace()
        att_weights_prem = F.softmax(align, dim=2) # (batch_size, max_seqlen_p, max_seqlen_h)
        att_weights_hypo = F.softmax(align, dim=1).permute(0,2,1) # (batch_size, max_seqlen_h, max_seqlen_p)

        hypo_encoded = hypo_encoded.transpose(1,2) # (batch_size, max_seqlen_h, 2*hidden_size)

        prem_weight_sum = torch.bmm(att_weights_prem, hypo_encoded) # (batch_size, max_seqlen_p, 2*hidden_size)
        hypo_weight_sum = torch.bmm(att_weights_hypo, prem_encoded) # (batch_size, max_seqlen_h, 2*hidden_size)

        prem_weight_sum.masked_fill(prem_ws_mask == 0, 0)
        hypo_weight_sum.masked_fill(hypo_ws_mask == 0, 0)

        return prem_weight_sum, hypo_weight_sum

    def encode(self, prem_embed, premlen, hypo_embed, hypolen):
        assert len(prem_embed.size()), len(hypo_embed.size())==(3, 3)
        prem_packed = pack_padded_sequence(prem_embed, premlen, batch_first=True, enforce_sorted=False)
        hypo_packed = pack_padded_sequence(hypo_embed, hypolen, batch_first=True, enforce_sorted=False)
        prem_lstm_out, _ = self.biLSTM(prem_packed)
        hypo_lstm_out, _ = self.biLSTM(hypo_packed) 
        prem_encoded, hypo_encoded = pad_packed_sequence(prem_lstm_out, batch_first=True)[0],  \
                                pad_packed_sequence(hypo_lstm_out, batch_first=True)[0] # (batch_size, max_seqlen, embedding_size)
        return prem_encoded, hypo_encoded
    
    def get_mask(self, prem, hypo, prem_encoded, hypo_encoded):
        prem_mask = (prem != 1).long()  # (batch_size, max_seqlen_p)
        hypo_mask = (hypo != 1).long()  # (batch_size, max_seqlen_h)

        prem_ws_mask = prem_mask.unsqueeze(2).expand_as(prem_encoded)
        hypo_ws_mask = hypo_mask.unsqueeze(2).expand_as(hypo_encoded)

        mask = torch.bmm(prem_mask.unsqueeze(2).float(),
                        hypo_mask.unsqueeze(1).float())    
        return mask, prem_ws_mask, hypo_ws_mask

    def _init_hidden(self):
        return torch.zeros((2 if self.bidirectional else 1),     \
                self.batch_size, self.hidden_size).to(self.device)
    
    def load_embedding(self, vectors):
        """Load pre-trained word embeddings:
        Arguments:
            vectors {torch.Tensor} -- pre-trained vectors. 
        """
        self.embedding.load(vectors)
