import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, fc1_size, output_size, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.rnn1 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((hidden_size * 2, ))
        self.bn = nn.BatchNorm1d(hidden_size * 4)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def rnn_flow(self, x, lengths):
        batch_size = lengths.size(0)
        h1, h2 = self.extract_features(x, lengths, self.rnn1, self.rnn2, self.layer_norm)
        h = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def mask2length(self, mask):
        ''' mask [batch_size, seq_length, feat_size]
        '''
        _mask = torch.mean(mask, dim=-1).long() # [batch_size, seq_len]
        length = torch.sum(_mask, dim=-1)       # [batch_size,]
        return length 

    def forward(self, x, mask):
        lengths = self.mask2length(mask)
        h = self.rnn_flow(x, lengths)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o, h

class SimpleClassifier(nn.Module):
    ''' Linear classifier, use embedding as input
        Linear approximation, should append with softmax
    '''
    def __init__(self, embd_size, output_dim, dropout):
        super(SimpleClassifier, self).__init__()
        self.dropout = dropout
        self.C = nn.Linear(embd_size, output_dim)
        self.dropout_op = nn.Dropout(dropout)

    def forward(self, x):
        if self.dropout > 0:
            x = self.dropout_op(x)
        return self.C(x)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3, use_bn=False):
        ''' Fully Connect classifier
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            output_dim: output feature dim
            activation: activation function
            dropout: dropout rate
        '''
        super().__init__()
        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        
        if len(layers) == 0:
            layers.append(input_dim)
            self.all_layers.append(Identity())
        
        self.fc_out = nn.Linear(layers[-1], output_dim)
        self.module = nn.Sequential(*self.all_layers)
    
    def forward(self, x):
        feat = self.module(x)
        out = self.fc_out(feat)
        return out, feat
    

class FcClassifier3(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.0, use_bn=False, 
                 num_extra_layers=2, layer_expansion_factor=1.5):
        super().__init__()
        
        # Enhanced layer configuration - add more layers
        enhanced_layers = []
        current_dim = input_dim
        
        # Add original specified layers
        for layer_dim in layers:
            enhanced_layers.append(layer_dim)
        
        # Add extra layers with progressive dimension reduction
        if len(enhanced_layers) > 0:
            last_dim = enhanced_layers[-1]
            for i in range(num_extra_layers):
                # Gradually reduce dimensions
                new_dim = max(output_dim * 2, int(last_dim / layer_expansion_factor))
                enhanced_layers.append(new_dim)
                last_dim = new_dim
        else:
            # If no layers specified, create default architecture
            enhanced_layers = [
                input_dim // 2,
                input_dim // 4,
                max(output_dim * 4, input_dim // 8),
                max(output_dim * 2, input_dim // 16)
            ]
        
        # Build the network layers
        all_layers = []
        current_dim = input_dim
        
        for i, layer_dim in enumerate(enhanced_layers):
            # Linear layer
            all_layers.append(nn.Linear(current_dim, layer_dim))
            
            # Activation function
            all_layers.append(nn.ReLU())
            
            # Optional batch normalization
            if use_bn:
                all_layers.append(nn.BatchNorm1d(layer_dim))
            
            # NO DROPOUT - removed as requested
            # if dropout > 0:
            #     all_layers.append(nn.Dropout(dropout))
            
            current_dim = layer_dim
        
        # Handle empty layers case
        if len(enhanced_layers) == 0:
            enhanced_layers = [input_dim]
            all_layers.append(nn.Identity())
            current_dim = input_dim
        
        # Create the sequential module
        self.module = nn.Sequential(*all_layers)
        
        # Output layer
        self.fc_out = nn.Linear(current_dim, output_dim)
        
        # Store layer information for debugging
        self.layer_dims = [input_dim] + enhanced_layers + [output_dim]
        
    def forward(self, x):
        feat = self.module(x)
        out = self.fc_out(feat)
        return out, feat
    
    def get_layer_info(self):
        """Helper method to print layer architecture"""
        print("FcClassifier Architecture:")
        for i, dim in enumerate(self.layer_dims[:-1]):
            print(f"  Layer {i}: {dim} -> {self.layer_dims[i+1]}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters())}")

    
    
    
class FcClassifier2(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3, use_bn=False, activation=nn.LeakyReLU(0.01)):
        ''' 
        Fully Connected classifier with improved block structure for enhanced training stability.
        
        Parameters:
        --------------------------
        input_dim: int, input feature dimension.
        layers: list of ints, e.g., [x1, x2, x3] will create 3 hidden layers with x1, x2, and x3 nodes.
        output_dim: int, output feature dimension (number of classes).
        dropout: float, dropout rate.
        use_bn: bool, whether to use Batch Normalization.
        activation: nn.Module, activation function (default is LeakyReLU).
        '''
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Build the hidden layers sequentially
        for hidden_dim in layers:
            block = []
            # Linear transformation
            block.append(nn.Linear(prev_dim, hidden_dim))
            # Batch Normalization (if enabled)
            if use_bn:
                block.append(nn.BatchNorm1d(hidden_dim))
            # Activation function
            block.append(activation)
            # Dropout (if dropout rate > 0)
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            # Append the entire block as a sequential module
            self.hidden_layers.append(nn.Sequential(*block))
            prev_dim = hidden_dim
        
        # If no hidden layers are specified, add an identity mapping to preserve dimensions.
        if len(layers) == 0:
            self.hidden_layers.append(nn.Identity())
            prev_dim = input_dim
        
        # Output layer: maps the final hidden representation to the desired number of classes.
        self.fc_out = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights for better convergence
        self.init_weights()
    
    def init_weights(self):
        # Use Kaiming initialization for linear layers and standard initialization for BN if used.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Pass input through all hidden blocks
        for layer in self.hidden_layers:
            x = layer(x)
        feat = x  # final feature representation before classification
        out = self.fc_out(feat)
        return out, feat


class EF_model_AL(nn.Module):
    def __init__(self, fc_classifier, lstm_classifier, out_dim_a, out_dim_v, fusion_size, num_class, dropout):
        ''' Early fusion model classifier
            Parameters:
            --------------------------
            fc_classifier: acoustic classifier
            lstm_classifier: lexical classifier
            out_dim_a: fc_classifier output dim
            out_dim_v: lstm_classifier output dim
            fusion_size: output_size for fusion model
            num_class: class number
            dropout: dropout rate
        '''
        super(EF_model_AL, self).__init__()
        self.fc_classifier = fc_classifier
        self.lstm_classifier = lstm_classifier
        self.out_dim = out_dim_a + out_dim_v
        self.dropout = nn.Dropout(dropout)
        self.num_class = num_class
        self.fusion_size = fusion_size
        # self.out = nn.Sequential(
        #     nn.Linear(self.out_dim, self.fusion_size),
        #     nn.ReLU(),
        #     nn.Linear(self.fusion_size, self.num_class),
        # )
        self.out1 = nn.Linear(self.out_dim, self.fusion_size)
        self.relu = nn.ReLU()
        self.out2 = nn.Linear(self.fusion_size, self.num_class)

    def forward(self, A_feat, L_feat, L_mask):
        _, A_out = self.fc_classifier(A_feat)
        _, L_out = self.lstm_classifier(L_feat, L_mask)
        feat = torch.cat([A_out, L_out], dim=-1)
        feat = self.dropout(feat)
        feat = self.relu(self.out1(feat))
        out = self.out2(self.dropout(feat))
        return out, feat
    

class MaxPoolFc(nn.Module):
    def __init__(self, hidden_size, num_class=4):
        super(MaxPoolFc, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_class),
            nn.ReLU()
        )
    
    def forward(self, x):
        ''' x shape => [batch_size, seq_len, hidden_size]
        '''
        batch_size, seq_len, hidden_size = x.size()
        x = x.view(batch_size, hidden_size, seq_len)
        # print(x.size())
        out = torch.max_pool1d(x, kernel_size=seq_len)
        out = out.squeeze()
        out = self.fc(out)
        
        return out


class Fusion(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3):
        super().__init__()
        self.fusion = nn.Sequential()
        for i in range(len(layers)):
            self.fusion.add_module(f'fusion_layer_{i}', nn.Linear(in_features=input_dim,
                                                               out_features=layers[i]))
            self.fusion.add_module(f'fusion_layer_{i}_dropout', nn.Dropout(dropout))
            self.fusion.add_module(f'fusion_layer_{i}_activation', nn.ReLU())
            input_dim = layers[i]

        self.fusion.add_module('fusion_layer_final',
                               nn.Linear(in_features=layers[-1], out_features=output_dim))

    def forward(self, x):
        feat = []
        out = self.fusion(x)
        return out, feat


if __name__ == '__main__':
    a = FcClassifier(256, [128], 4)
    print(a)