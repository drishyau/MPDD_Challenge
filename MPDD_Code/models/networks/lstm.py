import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=False):
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=bidirectional)
        assert embd_method in ['maxpool', 'attention', 'last', 'dense']
        self.embd_method = embd_method
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential()
            self.bidirectional = bidirectional
            if bidirectional:
                self.dense_layer.add_module('linear', nn.Linear(2 * self.hidden_size, self.hidden_size))
            else:
                self.dense_layer.add_module('linear', nn.Linear(self.hidden_size, self.hidden_size))
            self.dense_layer.add_module('activate', nn.Tanh())
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
       
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        attended_r_out = r_out * atten_weight  # 保持 [batch_size, seq_len, hidden_size]
        return attended_r_out  # No sum over time dimension

    def embd_maxpool(self, r_out, h_n):
      
        pooled_out, _ = torch.max(r_out, dim=1, keepdim=True)  # Keeps time dim
        return pooled_out.expand_as(r_out)  # Duplicate across time dimension

    def embd_last(self, r_out, h_n):
        
        return r_out  # Returns [batch_size, seq_len, hidden_size]

    def embd_dense(self, r_out, h_n):
      
        r_out = r_out.view(-1, r_out.size(2))  # Flatten to [batch_size * seq_len, hidden_size]
        dense_out = self.dense_layer(r_out)
        return dense_out.view(-1, r_out.size(1), self.hidden_size)  # Reshape back to [batch_size, seq_len, hidden_size]

    def forward(self, x):
       
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_' + self.embd_method)(r_out, h_n)
        return embd


# class LSTMEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=False, 
#                  num_layers=2, dropout=0.3):
#         super(LSTMEncoder, self).__init__()
        
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
        
#         # Multi-layer LSTM with dropout
#         self.rnn = nn.LSTM(
#             self.input_size, 
#             self.hidden_size, 
#             num_layers=num_layers,
#             batch_first=True, 
#             bidirectional=bidirectional,
#             dropout=dropout if num_layers > 1 else 0
#         )
        
#         self.embd_method = embd_method
        
#         # Layer normalization for LSTM outputs
#         lstm_output_size = hidden_size * (2 if bidirectional else 1)
#         self.layer_norm = nn.LayerNorm(lstm_output_size)
        
#         assert embd_method in ['maxpool', 'attention', 'last', 'dense']
        
#         if self.embd_method == 'attention':
#             self.attention_vector_weight = nn.Parameter(torch.Tensor(lstm_output_size, 1))
#             self.attention_layer = nn.Sequential(
#                 nn.Linear(lstm_output_size, lstm_output_size),
#                 nn.Tanh(),
#             )
#             self.softmax = nn.Softmax(dim=-1)
#             nn.init.xavier_uniform_(self.attention_vector_weight)
            
#         elif self.embd_method == 'dense':
#             self.dense_layer = nn.Sequential()
#             if bidirectional:
#                 self.dense_layer.add_module('linear', nn.Linear(2 * self.hidden_size, self.hidden_size))
#             else:
#                 self.dense_layer.add_module('linear', nn.Linear(self.hidden_size, self.hidden_size))
#             self.dense_layer.add_module('activate', nn.Tanh())

#     def embd_attention(self, r_out, h_n):
#         r_out = self.layer_norm(r_out)
#         hidden_reps = self.attention_layer(r_out)
#         atten_weight = (hidden_reps @ self.attention_vector_weight)
#         atten_weight = self.softmax(atten_weight)
#         attended_r_out = r_out * atten_weight
#         return attended_r_out

#     def embd_maxpool(self, r_out, h_n):
#         r_out = self.layer_norm(r_out)
#         pooled_out, _ = torch.max(r_out, dim=1, keepdim=True)
#         return pooled_out.expand_as(r_out)

#     def embd_last(self, r_out, h_n):
#         r_out = self.layer_norm(r_out)
#         return r_out

#     def embd_dense(self, r_out, h_n):
#         r_out = self.layer_norm(r_out)
#         batch_size, seq_len, feature_size = r_out.shape
#         r_out = r_out.view(-1, feature_size)
#         dense_out = self.dense_layer(r_out)
#         return dense_out.view(batch_size, seq_len, self.hidden_size)

#     def forward(self, x):
#         r_out, (h_n, h_c) = self.rnn(x)
#         embd = getattr(self, 'embd_' + self.embd_method)(r_out, h_n)
#         return embd