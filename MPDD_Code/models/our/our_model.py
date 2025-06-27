import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import *
from models.utils.config import OptConfig
import math
import torch.nn as nn



class Focal_Loss(torch.nn.Module):
    def __init__(self, weight=1, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='mean')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")

        
        

############################################################################# BASELINE MODEL ################################################################################
        
class ourModel(BaseModel, nn.Module):

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        nn.Module.__init__(self)
        super().__init__(opt)

        
        self.loss_names = []
        self.model_names = []

        # acoustic model
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.model_names.append('EmoA')

        # visual model
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.model_names.append('EmoV')

        # Transformer Fusion model
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head), batch_first=True)
        self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # Classifier
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # cls_input_size = 5*opt.hidden_size, same with max-len
        cls_input_size = opt.feature_max_len * opt.hidden_size + 1024  # with personalized feature

                                            
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        self.temperature = opt.temperature
        self.criterion_ce = torch.nn.CrossEntropyLoss()

        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = torch.nn.CrossEntropyLoss() 
            else:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=float(opt.lr), betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        # modify save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  


    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):

        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)

        self.emo_label = input['emo_label'].to(self.device)

        if 'personalized_feat' in input:
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None  # if no personalized features given
            

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)
        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        emo_feat_A = self.netEmoA(self.acoustic)
        emo_feat_V = self.netEmoV(self.visual)

        '''insure time dimension modification'''
        emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1) # (batch_size, seq_len, 2 * embd_size)
        
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        
        '''dynamic acquisition of bs'''
        batch_size = emo_fusion_feat.size(0)

        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)  # turn into [batch_size, feature_dim] 1028

        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)  # [batch_size, seq_len * feature_dim + 1024]

        '''for back prop'''
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        """-----------"""

        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label) 
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE

        loss.backward()

        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()

        self.optimizer.step()
       
        
        
##################################################################################  BEST Model ##########################################################################################



class CNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embd_method='last', 
                 kernel_sizes=[3, 5, 7], num_filters=64, num_layers=3, dropout=0.1):
        super(CNNEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_method = embd_method
        self.num_layers = num_layers
        
        # Multi-scale 1D CNN layers for sequential data
        self.conv_layers = nn.ModuleList()
        
        # First layer - from input to hidden
        conv_layer = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv_layers.append(conv_layer)
        
        # Intermediate layers
        for i in range(1, num_layers-1):
            kernel_idx = i % len(kernel_sizes)
            conv_layer = nn.Sequential(
                nn.Conv1d(num_filters, num_filters, kernel_size=kernel_sizes[kernel_idx], 
                         padding=kernel_sizes[kernel_idx]//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_layer)
        
        
        final_conv = nn.Sequential(
            nn.Conv1d(num_filters, hidden_size * 2, kernel_size=kernel_sizes[-1], 
                     padding=kernel_sizes[-1]//2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU()
        )
        self.conv_layers.append(final_conv)
        
        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Embedding method implementations
        assert embd_method in ['maxpool', 'attention', 'last', 'dense']
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)
            nn.init.xavier_uniform_(self.attention_vector_weight)
            
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.Tanh()
            )

    def embd_attention(self, conv_out):
        # conv_out: [batch, seq, hidden*2]
        conv_out = self.layer_norm(conv_out)
        hidden_reps = self.attention_layer(conv_out)
        atten_weight = (hidden_reps @ self.attention_vector_weight).squeeze(-1)  
        atten_weight = self.softmax(atten_weight).unsqueeze(-1)  
        attended_out = conv_out * atten_weight
        return attended_out


    def embd_maxpool(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        pooled_out, _ = torch.max(conv_out, dim=1, keepdim=True)
        return pooled_out.expand_as(conv_out)

    def embd_last(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        return conv_out

    def embd_dense(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        batch_size, seq_len, feature_size = conv_out.shape
        conv_out = conv_out.view(-1, feature_size)
        dense_out = self.dense_layer(conv_out)
        return dense_out.view(batch_size, seq_len, feature_size)  

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        # Conv1d expects [batch_size, channels, seq_len]
        x = x.transpose(1, 2)  # [batch_size, input_size, seq_len]
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Transpose back to [batch_size, seq_len, hidden_size*2]
        x = x.transpose(1, 2)
        
        # Apply embedding method
        embd = getattr(self, 'embd_' + self.embd_method)(x)
        return embd


class ourModel_Modified(BaseModel, nn.Module):

    def __init__(self, opt):
        nn.Module.__init__(self)
        super().__init__(opt)

        self.loss_names = []
        self.model_names = []

        self.netEmoA = CNNEncoder(
            opt.input_dim_a, 
            opt.embd_size_a, 
            embd_method=opt.embd_method_a,
            kernel_sizes=[3, 5, 7],  # Multi-scale kernels
            num_filters=128,         # More filters for better feature extraction
            num_layers=7,            # Deep CNN
            dropout=0.1
        )
        self.model_names.append('EmoA')

        self.netEmoV = CNNEncoder(
            opt.input_dim_v, 
            opt.embd_size_v, 
            embd_method=opt.embd_method_v,
            kernel_sizes=[3, 5, 7],  # Multi-scale kernels
            num_filters=128,         # More filters for better feature extraction
            num_layers=7,            # Deep CNN
            dropout=0.1
        )
        self.model_names.append('EmoV')


        audio_output_dim = opt.embd_size_a * 2
        visual_output_dim = opt.embd_size_v * 2
        concat_dim = audio_output_dim + visual_output_dim
        
        # Projection layer to match transformer expected dimension
        self.netFusionProjection = nn.Linear(concat_dim, opt.hidden_size)
        self.model_names.append('FusionProjection')

        # Enhanced Transformer Fusion model
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=opt.hidden_size,
            nhead=int(opt.Transformer_head), 
            dim_feedforward=opt.hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.netEmoFusion = torch.nn.TransformerEncoder(
            emo_encoder_layer, 
            num_layers=opt.Transformer_layers
        )
        self.model_names.append('EmoFusion')

        self.fusion_layer_norm = nn.LayerNorm(opt.hidden_size)

        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.feature_max_len * opt.hidden_size + 1024

        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        self.temperature = opt.temperature

        if self.isTrain:
            self.criterion_ce = nn.CrossEntropyLoss()
            self.criterion_focal = Focal_Loss()
            self.criterion_ce_regular = torch.nn.CrossEntropyLoss()
            

            paremeters = []
            for net in self.model_names:
                paremeters.append({'params': getattr(self, 'net' + net).parameters()})
            
            self.optimizer = torch.optim.AdamW(
                paremeters, 
                lr=float(opt.lr), 
                betas=(opt.beta1, 0.999),
                weight_decay=1e-4
            )
            self.optimizers.append(self.optimizer)
            
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        # modify save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # Forward method remains the same
    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)
        
        # Enhanced feature extraction with CNN
        emo_feat_A = self.netEmoA(self.acoustic)  # [batch, seq, embd_size_a*2]
        emo_feat_V = self.netEmoV(self.visual)    # [batch, seq, embd_size_v*2]

        # Concatenate features
        emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1)  # [batch, seq, concat_dim]
        
        # PROJECT to match transformer expected dimension
        emo_fusion_feat = self.netFusionProjection(emo_fusion_feat)  # [batch, seq, hidden_size]
        emo_fusion_feat = self.fusion_layer_norm(emo_fusion_feat)
        
        # Enhanced transformer fusion
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        
        # Reshape for classification
        batch_size = emo_fusion_feat.size(0)
        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)

        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)

        # Classification
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)



    def backward(self):
        """Enhanced backward pass with label smoothing and consistency loss"""
        # Label smoothing loss
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        
        # Focal loss
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        
        # Add consistency loss between classifiers
        consistency_loss = F.kl_div(
            F.log_softmax(self.emo_logits, dim=1),
            F.softmax(self.emo_logits_fusion, dim=1),
            reduction='batchmean'
        )
        
        # Combined loss
        total_loss = self.loss_emo_CE + self.loss_EmoF_CE + 0.1 * consistency_loss
        total_loss.backward()

        # Enhanced gradient clipping
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 0.5)

    def optimize_parameters(self, epoch):
        """Enhanced optimization with scheduler"""
        # Forward pass
        self.forward()
        
        # Backward pass
        self.optimizer.zero_grad()
        self.backward()
        
        # Update weights
        self.optimizer.step()
        
        # # Update learning rate
        # self.scheduler.step()

    # Keep other methods unchanged
    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def set_input(self, input):
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)

        if 'personalized_feat' in input:
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None






###########################################################################################################################################################################

class CNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embd_method='last', 
                 kernel_sizes=[3, 5, 7], num_filters=64, num_layers=3, dropout=0.1):
        super(CNNEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_method = embd_method
        self.num_layers = num_layers
        
        # Multi-scale 1D CNN layers for sequential data
        self.conv_layers = nn.ModuleList()
        
        # First layer - from input to hidden
        conv_layer = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv_layers.append(conv_layer)
        
        # Intermediate layers
        for i in range(1, num_layers-1):
            kernel_idx = i % len(kernel_sizes)
            conv_layer = nn.Sequential(
                nn.Conv1d(num_filters, num_filters, kernel_size=kernel_sizes[kernel_idx], 
                         padding=kernel_sizes[kernel_idx]//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_layer)
        
        
        final_conv = nn.Sequential(
            nn.Conv1d(num_filters, hidden_size * 2, kernel_size=kernel_sizes[-1], 
                     padding=kernel_sizes[-1]//2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU()
        )
        self.conv_layers.append(final_conv)
        
        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Embedding method implementations
        assert embd_method in ['maxpool', 'attention', 'last', 'dense']
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)
            nn.init.xavier_uniform_(self.attention_vector_weight)
            
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.Tanh()
            )

    def embd_attention(self, conv_out):
        # conv_out: [batch, seq, hidden*2]
        conv_out = self.layer_norm(conv_out)
        hidden_reps = self.attention_layer(conv_out)
        atten_weight = (hidden_reps @ self.attention_vector_weight).squeeze(-1)  
        atten_weight = self.softmax(atten_weight).unsqueeze(-1) 
        attended_out = conv_out * atten_weight
        return attended_out


    def embd_maxpool(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        pooled_out, _ = torch.max(conv_out, dim=1, keepdim=True)
        return pooled_out.expand_as(conv_out)

    def embd_last(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        return conv_out

    def embd_dense(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        batch_size, seq_len, feature_size = conv_out.shape
        conv_out = conv_out.view(-1, feature_size)
        dense_out = self.dense_layer(conv_out)
        return dense_out.view(batch_size, seq_len, feature_size) 


    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        # Conv1d expects [batch_size, channels, seq_len]
        x = x.transpose(1, 2)  # [batch_size, input_size, seq_len]
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Transpose back to [batch_size, seq_len, hidden_size*2]
        x = x.transpose(1, 2)
        
        # Apply embedding method
        embd = getattr(self, 'embd_' + self.embd_method)(x)
        return embd
    
    
    
     
class ourModelBaseModified(BaseModel, nn.Module):

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        nn.Module.__init__(self)
        super().__init__(opt)

        
        self.loss_names = []
        self.model_names = []

        self.netEmoA = CNNEncoder(
            opt.input_dim_a, 
            opt.embd_size_a, 
            embd_method=opt.embd_method_a,
            kernel_sizes=[3, 5, 7],  # Multi-scale kernels
            num_filters=128,         # More filters for better feature extraction
            num_layers=7,            # Deep CNN
            dropout=0.1
        )
        self.model_names.append('EmoA')

        # Enhanced visual model with CNN instead of LSTM
        self.netEmoV = CNNEncoder(
            opt.input_dim_v, 
            opt.embd_size_v, 
            embd_method=opt.embd_method_v,
            kernel_sizes=[3, 5, 7],  # Multi-scale kernels
            num_filters=128,         # More filters for better feature extraction
            num_layers=7,            # Deep CNN
            dropout=0.1
        )
        self.model_names.append('EmoV')

        # Calculate actual concatenated feature dimension
        # CNN outputs: embd_size * 2 (to match bidirectional LSTM)
        audio_output_dim = opt.embd_size_a * 2
        visual_output_dim = opt.embd_size_v * 2
        concat_dim = audio_output_dim + visual_output_dim
        
        # Projection layer to match transformer expected dimension
        self.netFusionProjection = nn.Linear(concat_dim, opt.hidden_size)
        self.model_names.append('FusionProjection')

        # Enhanced Transformer Fusion model
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=opt.hidden_size,
            nhead=int(opt.Transformer_head), 
            dim_feedforward=opt.hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.netEmoFusion = torch.nn.TransformerEncoder(
            emo_encoder_layer, 
            num_layers=opt.Transformer_layers
        )
        self.model_names.append('EmoFusion')

        # Add layer normalization
        self.fusion_layer_norm = nn.LayerNorm(opt.hidden_size)

        # Classifier
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # cls_input_size = 5*opt.hidden_size, same with max-len
        cls_input_size = opt.feature_max_len * opt.hidden_size + 1024  # with personalized feature

                                            
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        self.temperature = opt.temperature
        self.criterion_ce = Focal_Loss()

        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = torch.nn.CrossEntropyLoss() 
            else:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.AdamW(paremeters, lr=float(opt.lr), betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        # modify save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  


    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):

        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)

        self.emo_label = input['emo_label'].to(self.device)

        if 'personalized_feat' in input:
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None  
            

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)
        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        emo_feat_A = self.netEmoA(self.acoustic)  # [batch, seq, embd_size_a * 2]
        emo_feat_V = self.netEmoV(self.visual)    # [batch, seq, embd_size_v * 2]

        '''insure time dimension modification'''
        emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1)  # [batch, seq, 512]
        

        emo_fusion_feat = self.netFusionProjection(emo_fusion_feat)  # [batch, seq, 256]
        emo_fusion_feat = self.fusion_layer_norm(emo_fusion_feat)     # Apply layer norm
        
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)  
        '''dynamic acquisition of bs'''
        batch_size = emo_fusion_feat.size(0)

        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)

        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)

        '''for back prop'''
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        """-----------"""

        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)


    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label) 
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE

        loss.backward()

        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()

        self.optimizer.step()
        
        
        
        
############################################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ConformerBlock(nn.Module):
    """Conformer block combining convolution and self-attention"""
    def __init__(self, d_model, nhead, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Feed-forward 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.mhsa = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_mhsa = nn.LayerNorm(d_model)
        
        # Convolution module - FIXED
        self.norm_conv = nn.LayerNorm(d_model)  # Apply LayerNorm before transpose
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=conv_kernel_size//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )
        
        # Feed-forward 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Feed-forward 1 (with residual)
        x = x + 0.5 * self.ff1(x)
        
        # Multi-head self-attention (with residual)
        x_norm = self.norm_mhsa(x)
        attn_out, _ = self.mhsa(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out
        
        # Convolution module (with residual) - FIXED
        x_norm = self.norm_conv(x)  # Apply LayerNorm BEFORE transpose
        x_conv = x_norm.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv)
        x = x + x_conv.transpose(1, 2)  # (B, T, D)
        
        # Feed-forward 2 (with residual)
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """Conformer encoder replacing BiLSTM"""
    def __init__(self, input_dim, d_model, num_layers=4, nhead=8, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, conv_kernel_size, dropout) 
            for _ in range(num_layers)
        ])
        
        self.output_dim = d_model
        
    def forward(self, x, mask=None):
        # Project input to model dimension
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Apply Conformer blocks
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class CrossModalAttention(nn.Module):
    """Cross-modal attention for better fusion"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.cross_attn_a2v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_v2a = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        
        self.ff_a = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.ff_v = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, audio_feat, visual_feat, audio_mask=None, visual_mask=None):
        # Cross-attention: audio attends to visual
        audio_enhanced, _ = self.cross_attn_a2v(
            audio_feat, visual_feat, visual_feat, 
            key_padding_mask=visual_mask
        )
        audio_feat = self.norm_a(audio_feat + audio_enhanced)
        audio_feat = audio_feat + self.ff_a(audio_feat)
        
        # Cross-attention: visual attends to audio
        visual_enhanced, _ = self.cross_attn_v2a(
            visual_feat, audio_feat, audio_feat,
            key_padding_mask=audio_mask
        )
        visual_feat = self.norm_v(visual_feat + visual_enhanced)
        visual_feat = visual_feat + self.ff_v(visual_feat)
        
        return audio_feat, visual_feat

class GatedFusion(nn.Module):
    """Gated fusion mechanism for multimodal features"""
    def __init__(self, d_model):
        super().__init__()
        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_v = nn.Linear(d_model * 2, d_model)
        
    def forward(self, audio_feat, visual_feat):
        # Concatenate features
        concat_feat = torch.cat([audio_feat, visual_feat], dim=-1)
        
        # Compute gates
        gate_a = torch.sigmoid(self.gate_a(concat_feat))
        gate_v = torch.sigmoid(self.gate_v(concat_feat))
        
        # Apply gates and combine
        fused = gate_a * audio_feat + gate_v * visual_feat
        return fused

class EnhancedMultimodalModel(BaseModel, nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        BaseModel.__init__(self, opt)  # Proper BaseModel initialization
        
        self.loss_names = []
        self.model_names = []
        
        # Enhanced encoders using Conformer
        self.netEmoA = ConformerEncoder(
            input_dim=opt.input_dim_a,
            d_model=opt.hidden_size,
            num_layers=4,
            nhead=8,
            dropout=0.1
        )
        self.model_names.append('EmoA')
        
        self.netEmoV = ConformerEncoder(
            input_dim=opt.input_dim_v,
            d_model=opt.hidden_size,
            num_layers=4,
            nhead=8,
            dropout=0.1
        )
        self.model_names.append('EmoV')
        
        # Cross-modal attention for better fusion
        self.cross_modal_attention = CrossModalAttention(
            d_model=opt.hidden_size,
            nhead=8,
            dropout=0.1
        )
        
        # Gated fusion mechanism
        self.gated_fusion = GatedFusion(opt.hidden_size)
        
        # Enhanced Transformer fusion
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=opt.hidden_size,
            nhead=int(opt.Transformer_head),
            dim_feedforward=opt.hidden_size * 4,
            dropout=opt.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.netEmoFusion = nn.TransformerEncoder(
            fusion_encoder_layer, 
            num_layers=opt.Transformer_layers
        )
        self.model_names.append('EmoFusion')
        
        # Attention-based pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(opt.hidden_size, opt.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(opt.hidden_size // 2, 1)
        )
        
        # Classifier setup
        if opt.use_personalized:
            cls_input_size = opt.hidden_size * 2 + 1024  # *2 for avg+max pooling
        else:
            cls_input_size = opt.hidden_size * 2
            
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        
        # Enhanced classifiers
        self.netEmoC = self._build_enhanced_classifier(cls_input_size, cls_layers, opt.emo_output_dim, opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = self._build_enhanced_classifier(cls_input_size, cls_layers, opt.emo_output_dim, opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        # Training setup
        self.temperature = opt.temperature
        self.criterion_ce = nn.CrossEntropyLoss()
        
        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_ce = nn.CrossEntropyLoss()
                self.criterion_focal = nn.CrossEntropyLoss() 
            else:
                self.criterion_ce = nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()  # Make sure this class exists
                
            # Collect all parameters
            parameters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            parameters.append({'params': self.cross_modal_attention.parameters()})
            parameters.append({'params': self.gated_fusion.parameters()})
            parameters.append({'params': self.attention_pool.parameters()})
            
            self.optimizer = torch.optim.AdamW(parameters, lr=float(opt.lr), betas=(opt.beta1, 0.999), weight_decay=1e-4)
            # from lion_pytorch import Lion
            # self.optimizer = Lion(parameters, lr=3e-5, weight_decay=1e-2)
            #self.optimizer = torch.optim.RMSprop(parameters, lr=1e-3, alpha=0.9)

            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def _build_enhanced_classifier(self, input_size, hidden_sizes, output_size, dropout):
        """Build enhanced classifier with residual connections"""
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        return nn.Sequential(*layers)
    
    def set_input(self, input):
        """Required abstract method: unpack data from dataset and apply preprocessing"""
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)
        
        if self.opt.use_personalized and ('personalized_feat' in input):
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None
    
    def attention_pooling(self, x, mask=None):
        """Attention-based pooling"""
        attn_weights = self.attention_pool(x)  # (batch, seq, 1)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch, hidden)
        return pooled

    def forward(self, acoustic_feat=None, visual_feat=None):
        """Required abstract method: produce intermediate results"""
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)

        # Encode features with Conformer
        emo_feat_A = self.netEmoA(self.acoustic)  # (batch, seq, hidden_size)
        emo_feat_V = self.netEmoV(self.visual)    # (batch, seq, hidden_size)

        # Cross-modal attention
        emo_feat_A_enhanced, emo_feat_V_enhanced = self.cross_modal_attention(
            emo_feat_A, emo_feat_V
        )
        
        # Gated fusion
        fusion = self.gated_fusion(emo_feat_A_enhanced, emo_feat_V_enhanced)
        
        # Transformer fusion
        emo_fusion_feat = self.netEmoFusion(fusion)
        
        # Multi-scale pooling
        pooled_avg = emo_fusion_feat.mean(dim=1)
        pooled_max = emo_fusion_feat.max(dim=1)[0]
        
        # Combine pooling strategies
        pooled = torch.cat([pooled_avg, pooled_max], dim=-1)

        if self.personalized is not None:
            pooled = torch.cat([pooled, self.personalized], dim=-1)

        # FIX: Remove the unpacking since classifiers return only logits
        self.emo_logits_fusion = self.netEmoCF(pooled)
        self.emo_logits = self.netEmoC(pooled)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)


    def backward(self):
        """Calculate losses and perform backward pass"""
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE
        loss.backward()
        
        # Gradient clipping
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)
        
        # Also clip gradients for additional modules
        torch.nn.utils.clip_grad_norm_(self.cross_modal_attention.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.gated_fusion.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.attention_pool.parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Required abstract method: calculate losses, gradients, and update network weights"""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def post_process(self):
        """Load pretrained weights if available"""
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])
        
        if self.isTrain and hasattr(self, 'pretrained_encoder'):
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            # Only load if the pretrained encoder has the same architecture
            try:
                self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
                self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
                self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))
            except:
                print("Could not load pretrained weights - architecture mismatch. Training from scratch.")
        
        
    

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.dropout(self.net(x))
        
        
        
        
###########################################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json
from collections import OrderedDict

# A simple BiLSTM encoder that returns the full sequence output.
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional

    def forward(self, x):
        # x: (batch, seq, input_dim)
        out, _ = self.lstm(x)
        # out: (batch, seq, hidden_dim * num_directions)
        return out

class ourModel2_BiLSTM(BaseModel, nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        super().__init__(opt)
        
        self.loss_names = []
        self.model_names = []
        
        # Acoustic model: use BiLSTM encoder.
        self.netEmoA = BiLSTMEncoder(opt.input_dim_a, opt.embd_size_a, num_layers=1, bidirectional=True)
        self.model_names.append('EmoA')
        
        # Visual model: use BiLSTM encoder.
        self.netEmoV = BiLSTMEncoder(opt.input_dim_v, opt.embd_size_v, num_layers=1, bidirectional=True)
        self.model_names.append('EmoV')
        
        # Determine the concatenated dimension from both encoders:
        # Each branch outputs dimension = 2 * opt.embd_size
        project_in_dim = 2 * opt.embd_size_a + 2 * opt.embd_size_v
        
        # Add a projection layer to map the concatenated features to opt.hidden_size,
        # so that the Transformer encoder receives the correct input dimension.
        self.fc_proj = nn.Linear(project_in_dim, opt.hidden_size)
        
        # Transformer Fusion model
        emo_encoder_layer = nn.TransformerEncoderLayer(
            d_model=opt.hidden_size, 
            nhead=int(opt.Transformer_head), 
            batch_first=True
        )
        self.netEmoFusion = nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')
        
        # Classifier Input Size: adjust based on whether personalized features are used.
        if opt.use_personalized:
            cls_input_size = opt.feature_max_len * opt.hidden_size + 1024
        else:
            cls_input_size = opt.feature_max_len * opt.hidden_size

        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        self.netEmoC = FcClassifier2(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier2(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        self.temperature = opt.temperature
        self.criterion_ce = nn.CrossEntropyLoss()

        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_ce = nn.CrossEntropyLoss()
                self.criterion_focal = nn.CrossEntropyLoss() 
            else:
                self.criterion_ce = nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()
            parameters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]            
            self.optimizer = torch.optim.Adam(parameters, lr=float(opt.lr), betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)  


    
    def post_process(self):
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])
        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def set_input(self, input):
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)
        if self.opt.use_personalized and ('personalized_feat' in input):
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)

        # Obtain sequence features from each branch.
        emo_feat_A = self.netEmoA(self.acoustic)  # shape: (batch, seq, 2*embd_size_a)
        emo_feat_V = self.netEmoV(self.visual)      # shape: (batch, seq, 2*embd_size_v)

        # Concatenate along the feature dimension.
        fusion = torch.cat((emo_feat_V, emo_feat_A), dim=-1)  # shape: (batch, seq, project_in_dim)
        # Project to the desired Transformer dimension.
        fusion = self.fc_proj(fusion)  # now shape: (batch, seq, opt.hidden_size)
        
        # Fuse with the Transformer encoder.
        emo_fusion_feat = self.netEmoFusion(fusion)
        batch_size = emo_fusion_feat.size(0)
        # Flatten the sequence dimension.
        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)
        
        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)
        
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        
        
        
###########################################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from collections import OrderedDict


class EnhancedAudioEncoderCNNBiLSTM(nn.Module):
    """
    Enhanced audio encoder using multiple CNN layers followed by a bidirectional LSTM.
    
    - A stack of convolutional blocks extracts more refined local temporal patterns.
    - A bidirectional LSTM captures long-term dependencies.
    """
    def __init__(self, input_dim, cnn_channels, kernel_size, hidden_size, 
                 num_conv_layers=2, num_layers=1, dropout=0.0, bidirectional=True):
        super(EnhancedAudioEncoderCNNBiLSTM, self).__init__()
        conv_layers = []
        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else cnn_channels
            conv_layers.append(nn.Conv1d(in_channels=in_channels, 
                                         out_channels=cnn_channels, 
                                         kernel_size=kernel_size, 
                                         padding=kernel_size // 2))
            conv_layers.append(nn.BatchNorm1d(cnn_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*conv_layers)
        # LSTM expects input of shape [batch, seq_len, channels]
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        self.output_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        cnn_out = self.cnn_layers(x)  # [batch, cnn_channels, seq_len]
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)  # [batch, seq_len, hidden_size * 2]
        return lstm_out

class EnhancedVisualEncoderCNN(nn.Module):
    """
    Enhanced visual encoder using multiple CNN layers.
    
    This module uses a stack of convolutional blocks to extract higher-level features
    from the visual input sequence.
    """
    def __init__(self, input_dim, cnn_channels, kernel_size, num_conv_layers=2, dropout=0.0):
        super(EnhancedVisualEncoderCNN, self).__init__()
        conv_layers = []
        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else cnn_channels
            conv_layers.append(nn.Conv1d(in_channels=in_channels, 
                                         out_channels=cnn_channels, 
                                         kernel_size=kernel_size, 
                                         padding=kernel_size // 2))
            conv_layers.append(nn.BatchNorm1d(cnn_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
        self.cnn_layers = nn.Sequential(*conv_layers)
        self.output_dim = cnn_channels

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        cnn_out = self.cnn_layers(x)  # [batch, cnn_channels, seq_len]
        cnn_out = cnn_out.permute(0, 2, 1)  # [batch, seq_len, cnn_channels]
        return cnn_out


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder to reconstruct the original input sequence from the encoded representation.
    """
    def __init__(self, encoded_size, output_dim, num_layers=1, dropout=0.0):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(encoded_size, encoded_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(encoded_size, output_dim)
        
    def forward(self, encoded_seq):
        outputs, _ = self.lstm(encoded_seq)
        recon = self.fc(outputs)
        return recon


class Model5(BaseModel, nn.Module):

    def __init__(self, opt):
        """
        Initialize the enhanced model with:
         - Audio: CNN + BiLSTM encoder with increased convolutional depth,
         - Visual: CNN-based encoder with increased convolutional depth,
         - Fusion via a Transformer,
         - Classifiers with more linear layers,
         - Autoencoder decoders for reconstruction.
        
        Parameters:
            opt (Option class): Contains experiment flags and hyperparameters.
        """
        nn.Module.__init__(self)
        super().__init__(opt)

        self.loss_names = []
        self.model_names = []

        # Use a default value of 1 for lstm_layers if not provided.
        lstm_layers = getattr(opt, "lstm_layers", 1)


        self.netEmoA = EnhancedAudioEncoderCNNBiLSTM(
            input_dim=opt.input_dim_a,
            cnn_channels=getattr(opt, "audio_cnn_channels", 64),
            kernel_size=getattr(opt, "audio_kernel_size", 3),
            hidden_size=getattr(opt, "audio_hidden_size", 128),
            num_conv_layers=getattr(opt, "num_audio_conv_layers", 3),  # default to 2 conv layers
            num_layers=lstm_layers,
            dropout=opt.dropout_rate,
            bidirectional=True
        )
        self.model_names.append('EmoA')


        self.netEmoV = EnhancedVisualEncoderCNN(
            input_dim=opt.input_dim_v,
            cnn_channels=getattr(opt, "visual_cnn_channels", 64),
            kernel_size=getattr(opt, "visual_kernel_size", 3),
            num_conv_layers=getattr(opt, "num_visual_conv_layers", 3),
            dropout=opt.dropout_rate
        )
        self.model_names.append('EmoV')

        # Store hidden_size for fusion projection.
        self.hidden_size = opt.hidden_size


        # Fuse the two modalities via a Transformer encoder.
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(opt.Transformer_head),
            batch_first=True
        )
        self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')


        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        # The classifier input is composed of the fused features plus optional personalized features.
        cls_input_size = opt.feature_max_len * self.hidden_size + 1024
        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')


        self.netDecA = LSTMDecoder(
            encoded_size=getattr(opt, "audio_hidden_size", 128) * 2,
            output_dim=opt.input_dim_a,
            num_layers=lstm_layers,
            dropout=opt.dropout_rate
        )
        self.model_names.append('DecA')

        # For visual: output dimension from visual encoder = visual_cnn_channels.
        self.netDecV = LSTMDecoder(
            encoded_size=getattr(opt, "visual_cnn_channels", 64),
            output_dim=opt.input_dim_v,
            num_layers=lstm_layers,
            dropout=opt.dropout_rate
        )
        self.model_names.append('DecV')
        self.loss_names.append('recon_A')
        self.loss_names.append('recon_V')

        self.temperature = opt.temperature
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.criterion_recon = torch.nn.MSELoss()

        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = torch.nn.CrossEntropyLoss()
            else:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = Focal_Loss()
            parameters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(parameters, lr=float(opt.lr), betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight
            self.recon_weight = getattr(opt, "recon_weight", 1.0)


        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def post_process(self):
        # Called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])
        if self.isTrain:
            print('[ Init ] Loading parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        # Transfer input tensors to device.
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)
        if self.opt.use_personalized and ('personalized_feat' in input):
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)



        encoded_A = self.netEmoA(self.acoustic)  # [batch, seq_len, audio_hidden_size*2]
        encoded_V = self.netEmoV(self.visual)    # [batch, seq_len, visual_cnn_channels]

        self.recon_A = self.netDecA(encoded_A)
        self.recon_V = self.netDecV(encoded_V)


        emo_fusion_feat = torch.cat((encoded_V, encoded_A), dim=-1)
        if emo_fusion_feat.shape[-1] != self.hidden_size:
            projection = nn.Linear(emo_fusion_feat.shape[-1], self.hidden_size)
            projection = projection.to(emo_fusion_feat.device)
            emo_fusion_feat = projection(emo_fusion_feat)
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        batch_size = emo_fusion_feat.size(0)
        emo_fusion_feat = emo_fusion_feat.permute(1, 0, 2).reshape(batch_size, -1)
        if self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def backward(self):
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        self.loss_recon_A = self.criterion_recon(self.recon_A, self.acoustic)
        self.loss_recon_V = self.criterion_recon(self.recon_V, self.visual)
        loss = self.loss_emo_CE + self.loss_EmoF_CE + self.recon_weight * (self.loss_recon_A + self.loss_recon_V)
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        
        
        
###########################################################################################################################################

class AttentiveStatsPooling(nn.Module):
    """Attentive Statistics Pooling layer"""
    def __init__(self, input_dim, attention_dim=128, global_context=True):
        super(AttentiveStatsPooling, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.global_context = global_context
        
        # Attention mechanism
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Global context vector (optional)
        if self.global_context:
            self.global_context_vector = nn.Parameter(torch.randn(input_dim))
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            lengths: actual sequence lengths for each batch item
        Returns:
            output: [batch_size, input_dim * 2] (concatenated mean and std)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Calculate attention weights
        if self.global_context:
            # Add global context to input features
            global_context = self.global_context_vector.unsqueeze(0).unsqueeze(0)
            global_context = global_context.expand(batch_size, seq_len, -1)
            attention_input = x + global_context
        else:
            attention_input = x
            
        # Compute attention scores
        attention_scores = self.attention_layer(attention_input).squeeze(-1)  # [batch, seq_len]
        
        # Apply length masking if provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attention_scores.masked_fill_(mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Calculate weighted mean
        weighted_mean = torch.sum(attention_weights * x, dim=1)  # [batch, input_dim]
        
        # Calculate weighted standard deviation
        squared_diff = (x - weighted_mean.unsqueeze(1)) ** 2
        weighted_variance = torch.sum(attention_weights * squared_diff, dim=1)
        weighted_std = torch.sqrt(weighted_variance + 1e-8)  # Add epsilon for numerical stability
        
        # Concatenate mean and std
        output = torch.cat([weighted_mean, weighted_std], dim=-1)  # [batch, input_dim * 2]
        
        return output, attention_weights.squeeze(-1)


class CNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embd_method='last', 
                 kernel_sizes=[3, 5, 7], num_filters=128, num_layers=5, dropout=0.1):
        super(CNNEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_method = embd_method
        self.num_layers = num_layers
        
        # Enhanced multi-scale 1D CNN layers
        self.conv_layers = nn.ModuleList()
        
        if self.embd_method == 'attentive_stats':
            self.attentive_stats_pooling = AttentiveStatsPooling(
                input_dim=hidden_size * 2,  # BiLSTM output dimension
                attention_dim=hidden_size,
                global_context=True
            )
        
        # First layer - from input to num_filters
        conv_layer = nn.Sequential(
            nn.Conv1d(input_size, num_filters, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv_layers.append(conv_layer)
        
        # Intermediate CNN layers with residual-like connections
        for i in range(1, num_layers-1):
            kernel_idx = i % len(kernel_sizes)
            conv_layer = nn.Sequential(
                nn.Conv1d(num_filters, num_filters, kernel_size=kernel_sizes[kernel_idx], 
                         padding=kernel_sizes[kernel_idx]//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_layer)
        
        # BiLSTM layer for temporal modeling after CNN
        self.bilstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Final projection to match expected output dimension
        self.final_projection = nn.Linear(hidden_size * 2, hidden_size * 2)
        
        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Enhanced attention mechanism
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)
            nn.init.xavier_uniform_(self.attention_vector_weight)
            
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.Tanh()
            )

    def embd_attention(self, conv_out):
        # Enhanced attention with better normalization
        conv_out = self.layer_norm(conv_out)
        hidden_reps = self.attention_layer(conv_out)
        atten_weight = (hidden_reps @ self.attention_vector_weight).squeeze(-1)
        atten_weight = self.softmax(atten_weight).unsqueeze(-1)
        attended_out = conv_out * atten_weight
        return attended_out

    def embd_maxpool(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        pooled_out, _ = torch.max(conv_out, dim=1, keepdim=True)
        return pooled_out.expand_as(conv_out)

    def embd_last(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        return conv_out

    def embd_dense(self, conv_out):
        conv_out = self.layer_norm(conv_out)
        batch_size, seq_len, feature_size = conv_out.shape
        conv_out = conv_out.view(-1, feature_size)
        dense_out = self.dense_layer(conv_out)
        return dense_out.view(batch_size, seq_len, feature_size)
    
    def embd_attentive_stats(self, conv_out):
        """Attentive statistics pooling method"""
        pooled_output, attention_weights = self.attentive_stats_pooling(conv_out)
        # Expand to match sequence length for consistency with other methods
        batch_size, seq_len, _ = conv_out.shape
        return pooled_output.unsqueeze(1).expand(-1, seq_len, -1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        # Conv1d expects [batch_size, channels, seq_len]
        x = x.transpose(1, 2)  # [batch_size, input_size, seq_len]
        
        # Apply CNN layers for spatial feature extraction
        # CORRECT - Apply all conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        
        # Transpose back for BiLSTM: [batch_size, seq_len, num_filters]
        x = x.transpose(1, 2)
        
        # Apply BiLSTM for temporal modeling
        lstm_out, (hidden, cell) = self.bilstm(x)
        
        # Final projection and normalization
        lstm_out = self.final_projection(lstm_out)
        
        # Apply embedding method
        embd = getattr(self, 'embd_' + self.embd_method)(lstm_out)
        return embd


class Model23(BaseModel, nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        super().__init__(opt)

        self.loss_names = []
        self.model_names = []

        # Enhanced acoustic model with CNN-BiLSTM architecture
        self.netEmoA = CNNEncoder(
            opt.input_dim_a, 
            opt.embd_size_a, 
            embd_method='attentive_stats',  # Use attentive stats pooling
            kernel_sizes=[3, 5, 7, 9],
            num_filters=256,
            num_layers=6,
            dropout=0.15
        )
        
        # Enhanced visual model with Attentive Stats Pooling
        self.netEmoV = CNNEncoder(
            opt.input_dim_v, 
            opt.embd_size_v, 
            embd_method='attentive_stats',  # Use attentive stats pooling
            kernel_sizes=[3, 5, 7, 9],
            num_filters=256,
            num_layers=6,
            dropout=0.15
        )
        
        # Update feature dimensions (now 4x due to mean+std concatenation)
        audio_output_dim = opt.embd_size_a * 4  # 2 (BiLSTM) * 2 (mean+std)
        visual_output_dim = opt.embd_size_v * 4  # 2 (BiLSTM) * 2 (mean+std)
        concat_dim = audio_output_dim + visual_output_dim

        
        # Enhanced projection layer with better initialization
        self.netFusionProjection = nn.Sequential(
            nn.Linear(concat_dim, opt.hidden_size),
            nn.LayerNorm(opt.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.model_names.append('FusionProjection')

        # Enhanced Transformer with better configuration
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=opt.hidden_size,
            nhead=max(8, int(opt.Transformer_head)),  # Ensure minimum 8 heads
            dim_feedforward=opt.hidden_size * 6,      # Larger feedforward for better capacity
            dropout=0.1,
            activation='gelu',                        # GELU activation for better performance
            batch_first=True,
            norm_first=True                          # Pre-layer normalization
        )
        self.netEmoFusion = torch.nn.TransformerEncoder(
            emo_encoder_layer, 
            num_layers=max(6, opt.Transformer_layers)  # Ensure minimum 6 layers
        )
        self.model_names.append('EmoFusion')

        # Enhanced layer normalization
        self.fusion_layer_norm = nn.LayerNorm(opt.hidden_size)

        # Classification layers with better architecture
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.feature_max_len * opt.hidden_size + 1024

        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        self.temperature = opt.temperature

        if self.isTrain:
            self.criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing
            self.criterion_focal = Focal_Loss()
            self.criterion_ce_regular = torch.nn.CrossEntropyLoss()
            
            # Enhanced optimizer configuration
            paremeters = []
            for net in self.model_names:
                paremeters.append({'params': getattr(self, 'net' + net).parameters()})
            
            self.optimizer = torch.optim.AdamW(
                paremeters, 
                lr=float(opt.lr), 
                betas=(0.9, 0.999),           # Better beta values
                weight_decay=1e-4,            # L2 regularization
                eps=1e-8                      # Numerical stability
            )
            self.optimizers.append(self.optimizer)
            
            # Learning rate scheduler for better convergence
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
            
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        # Modify save_dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)
        
        # Enhanced CNN-BiLSTM feature extraction
        emo_feat_A = self.netEmoA(self.acoustic)  # [batch, seq, embd_size_a*2]
        emo_feat_V = self.netEmoV(self.visual)    # [batch, seq, embd_size_v*2]

        # Feature fusion with better alignment
        emo_fusion_feat = torch.cat((emo_feat_V, emo_feat_A), dim=-1)
        
        # Enhanced projection with normalization
        emo_fusion_feat = self.netFusionProjection(emo_fusion_feat)
        emo_fusion_feat = self.fusion_layer_norm(emo_fusion_feat)
        
        # Enhanced transformer processing
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat)
        
        # FIXED: Direct reshape without incorrect permutation
        batch_size = emo_fusion_feat.size(0)
        emo_fusion_feat = emo_fusion_feat.reshape(batch_size, -1)

        if hasattr(self, 'personalized') and self.personalized is not None:
            emo_fusion_feat = torch.cat((emo_fusion_feat, self.personalized), dim=-1)

        # Dual classification for ensemble effect
        self.emo_logits_fusion, _ = self.netEmoCF(emo_fusion_feat)
        self.emo_logits, _ = self.netEmoC(emo_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

        return self.emo_logits, self.emo_logits_fusion


    def backward(self):
        """Enhanced backward pass with NaN protection"""
        # Add epsilon to prevent log(0) in cross-entropy
        epsilon = 1e-7
        
        # Clamp logits to prevent extreme values
        self.emo_logits = torch.clamp(self.emo_logits, min=-10, max=10)
        self.emo_logits_fusion = torch.clamp(self.emo_logits_fusion, min=-10, max=10)
        
        # Main classification loss with numerical stability
        try:
            self.loss_emo_CE = self.ce_weight * self.criterion_ce(self.emo_logits, self.emo_label)
            
            # Check for NaN in CE loss
            if torch.isnan(self.loss_emo_CE):
                print("Warning: CE loss is NaN, using backup loss calculation")
                # Backup: use standard cross entropy without label smoothing
                self.loss_emo_CE = self.ce_weight * F.cross_entropy(self.emo_logits, self.emo_label)
        except:
            print("Error in CE loss calculation, using backup")
            self.loss_emo_CE = self.ce_weight * F.cross_entropy(self.emo_logits, self.emo_label)
        
        # Focal loss for handling class imbalance
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        
        # Consistency regularization with numerical stability
        log_p = F.log_softmax(self.emo_logits / self.temperature, dim=1)
        p_target = F.softmax(self.emo_logits_fusion / self.temperature, dim=1)
        
        # Add epsilon to prevent log(0)
        p_target = torch.clamp(p_target, min=epsilon, max=1-epsilon)
        
        consistency_loss = F.kl_div(log_p, p_target, reduction='batchmean') * (self.temperature ** 2)
        
        # Feature diversity loss with numerical stability
        logits_mean = F.normalize(self.emo_logits.mean(dim=0), dim=0)
        logits_fusion_mean = F.normalize(self.emo_logits_fusion.mean(dim=0), dim=0)
        
        feature_div_loss = -F.cosine_similarity(logits_mean, logits_fusion_mean, dim=0).mean()
        
        # Combined loss with NaN checking
        total_loss = (
            self.loss_emo_CE + 
            self.loss_EmoF_CE + 
            0.1 * consistency_loss +
            0.05 * feature_div_loss
        )
        
        # Final NaN check
        if torch.isnan(total_loss):
            print("Warning: Total loss is NaN, using only focal loss")
            total_loss = self.loss_EmoF_CE
        
        total_loss.backward()


    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        
        # Add gradient clipping to prevent NaN loss
        torch.nn.utils.clip_grad_norm_(
            [p for net_name in self.model_names 
            for p in getattr(self, 'net' + net_name).parameters()], 
            max_norm=1.0
        )
        
        self.optimizer.step()
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

    def post_process(self):
        """Enhanced post-processing with better weight loading"""
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain and hasattr(self, 'pretrained_encoder'):
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            
            # Load with strict=False to handle architecture differences
            try:
                self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()), strict=False)
                self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()), strict=False)
                self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()), strict=False)
                print('Successfully loaded pretrained weights')
            except Exception as e:
                print(f'Warning: Could not load some pretrained weights: {e}')

    def set_input(self, input):
        """Enhanced input handling with better device management"""
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)

        if 'personalized_feat' in input:
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None



