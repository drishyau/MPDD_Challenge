# from datetime import datetime
# import os,json,time,argparse,torch,yaml
# from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
# from torch.utils.data import DataLoader
# from train_val_split import train_val_split1, train_val_split2
# from models.our.our_model import *
# from dataset import *
# from utils.logger import get_logger
# import numpy as np


# def GetYAMLConfigs(path):
#     with open(path, 'r') as file:
#         c = yaml.safe_load(file)
#     return c

# class Opt:
#     def __init__(self, config_dict):
#         self.__dict__.update(config_dict)

# def load_config(config_file):
#     with open(config_file, 'r') as f:
#         return json.load(f)

# def eval(model, val_loader, device):
#     model.eval()
#     total_emo_pred = []
#     total_emo_label = []

#     with torch.no_grad():
#         for data in val_loader:
#             for k, v in data.items():
#                 data[k] = v.to(device)
#             model.set_input(data)
#             model.test()
#             emo_pred = model.emo_pred.argmax(dim=1).cpu().numpy()
#             emo_label = data['emo_label'].cpu().numpy()
#             total_emo_pred.append(emo_pred)
#             total_emo_label.append(emo_label)

#     total_emo_pred = np.concatenate(total_emo_pred)
#     total_emo_label = np.concatenate(total_emo_label)

#     emo_acc_unweighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=None)
#     class_counts = np.bincount(total_emo_label)  # Get the sample size for each category
#     sample_weights = 1 / (class_counts[total_emo_label] + 1e-6)  # Calculate weights for each sample to avoid division by zero errors
#     emo_acc_weighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=sample_weights)

#     emo_f1_weighted = f1_score(total_emo_label, total_emo_pred, average='weighted')
#     emo_f1_unweighted = f1_score(total_emo_label, total_emo_pred, average='macro')
#     emo_cm = confusion_matrix(total_emo_label, total_emo_pred)

#     return total_emo_label,total_emo_pred,emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm


# def train_model(train_json, model, audio_path='', video_path='', max_len=5,
#                 best_model_name='best_model.pth', seed=None):

#     logger.info(f"personalized features used: {configs['personalized_features_file']}")

#     num_epochs = configs['num_epochs']
#     device = configs['device']
#     print(f"device: {device}")
#     model.to(device)

#     # split training and validation set
#     # data = json.load(open(train_json, 'r'))
#     if configs['track_option']=='Track1':
#         train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=0.1, random_seed=seed)
#     elif configs['track_option']=='Track2':
#         train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=0.1,
#                                                                                      seed=seed)

    
#     train_loader = DataLoader(
#     AudioVisualDataset(train_data,configs['labelcount'], configs['personalized_features_file'], max_len=opt.feature_max_len, batch_size=configs['batch_size'],audio_path=audio_path, video_path=video_path, use_personalized=configs.get('use_personalized', True)
#     ), batch_size=configs['batch_size'], shuffle=True)

#     val_loader = DataLoader(
#     AudioVisualDataset(val_data,configs['labelcount'], configs['personalized_features_file'], max_len=opt.feature_max_len, batch_size=configs['batch_size'],audio_path=audio_path, video_path=video_path, use_personalized=configs.get('use_personalized', True)
#     ), batch_size=configs['batch_size'], shuffle=True)
    
#     logger.info('The number of training samples = %d' % len(train_loader.dataset))
#     logger.info('The number of val samples = %d' % len(val_loader.dataset))

#     best_emo_acc = 0.0
#     best_emo_f1 = 0.0
#     best_emo_epoch = 1
#     best_emo_cm = []

#     for epoch in range(num_epochs):
#         model.train(True)
#         total_loss = 0

#         for i, data in enumerate(train_loader):
#             for k, v in data.items():
#                 data[k] = v.to(device)
#             model.set_input(data)
#             model.optimize_parameters(epoch)

#             losses = model.get_current_losses() 
#             total_loss += losses['emo_CE']

        
        
#         # for i, data in enumerate(train_loader):
#         #     for k, v in data.items():
#         #         data[k] = v.to(device)
#         #     model.set_input(data)
#         #     model.optimize_parameters(epoch)

#         #     losses = model.get_current_losses()
#         #     total_loss += losses['emo_CE_A'] + losses['emo_CE_V'] + losses['emo_CE_combined']
            
            
#         avg_loss = total_loss / len(train_loader)


#         # evaluation
#         label, pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm = eval(model, val_loader,
#                                                                                                 device)

#         logger.info(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.10f}, "
#                     f"Weighted F1: {emo_f1_weighted:.10f}, Unweighted F1: {emo_f1_unweighted:.10f}, "
#                     f"Weighted Acc: {emo_acc_weighted:.10f}, Unweighted Acc: {emo_acc_unweighted:.10f}")
#         logger.info('Confusion Matrix:\n{}'.format(emo_cm))

#         if emo_f1_weighted > best_emo_f1:
#             cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
#             best_emo_f1 = emo_f1_weighted
#             best_emo_f1_unweighted = emo_f1_unweighted
#             best_emo_acc = emo_acc_weighted
#             best_emo_acc_unweighted = emo_acc_unweighted
#             best_emo_cm = emo_cm
#             best_emo_epoch = epoch + 1
#             best_model = model
#             save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name), best_model_name)
#             torch.save(model.state_dict(), save_path)
#             print("Saved best model.")

#     logger.info(f"Training complete. Random seed: {seed}. Best epoch: {best_emo_epoch}.")
#     logger.info(f"Best Weighted F1: {best_emo_f1:.4f}, Best Unweighted F1: {best_emo_f1_unweighted:.4f}, "
#                 f"Best Weighted Acc: {best_emo_acc:.4f}, Best Unweighted Acc: {best_emo_acc_unweighted:.4f}.")
#     logger.info('Confusion Matrix:\n{}'.format(best_emo_cm))

#     # output results to CSV
#     csv_file = f'{opt.log_dir}/{opt.name}.csv'
#     formatted_best_emo_cm = ' '.join([f"[{' '.join(map(str, row))}]" for row in best_emo_cm])
#     header = f"Time,random seed,splitwindow_time,labelcount,audiofeature_method,videofeature_method," \
#              f"batch_size,num_epochs,feature_max_len,lr," \
#              f"Weighted_F1,Unweighted_F1,Weighted_Acc,Unweighted_Acc,Confusion_Matrix"
#     result_value = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{seed},{configs['splitwindow_time']},{configs['labelcount']},{configs['audiofeature_method']},{configs['videofeature_method']}," \
#                f"{configs['batch_size']},{configs['num_epochs']},{opt.feature_max_len},{opt.lr:.6f}," \
#                f"{best_emo_f1:.4f},{best_emo_f1_unweighted:.4f},{best_emo_acc:.4f},{best_emo_acc_unweighted:.4f},{formatted_best_emo_cm}"

#     file_exists = os.path.exists(csv_file)
#     # Open file (append if file exists, create if it doesn't)
#     with open(csv_file, mode='a') as file:
#         if not file_exists:
#             file.write(header + '\n')
#         file.write(result_value + '\n')

#     return best_emo_f1, best_emo_f1_unweighted, best_emo_acc, best_emo_acc_unweighted, best_emo_cm


# if __name__ == '__main__':
#     P = argparse.ArgumentParser()
#     P.add_argument("--config", type=str, required=True)
#     A = P.parse_args()
#     configs = GetYAMLConfigs(path=A.config)

#     config = load_config('config.json')
#     opt = Opt(config)
#     opt.use_personalized = configs['use_personalized']
#     #opt.use_wav2vec = configs['use_wav2vec']
#     opt.emo_output_dim = configs['labelcount']
#     opt.feature_max_len = configs['feature_max_len']
#     opt.lr = float(configs['lr'])
#     opt.lr_policy = configs.get('lr_policy', 'step')
#     opt.lr_decay_step = configs.get('lr_decay_step', 10)
#     opt.lr_decay_gamma = configs.get('lr_decay_gamma', 0.1)
#     opt.class_distribution = [135,99,30]




#     configs['data_rootpath'] = "/home/hiddenrock/DepressionDetection/MPDD-Young"
    
#     audio_path = os.path.join(configs['data_rootpath'], 'Training', f"{configs['splitwindow_time']}", 'Audio', f"{configs['audiofeature_method']}") + '/'
#     video_path = os.path.join(configs['data_rootpath'], 'Training', f"{configs['splitwindow_time']}", 'Visual', f"{configs['videofeature_method']}") + '/'

#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Directory not found: {audio_path}")

#     for filename in os.listdir(audio_path):
#         if filename.endswith('.npy'):
#             opt.input_dim_a = np.load(audio_path + filename).shape[1]
#             break

#     for filename in os.listdir(video_path):
#         if filename.endswith('.npy'):
#             opt.input_dim_v = np.load(video_path + filename).shape[1]            
#             break
    

#     opt.name = f"{configs['splitwindow_time']}_{configs['labelcount']}labels_{configs['audiofeature_method']}+{configs['videofeature_method']}"

#     logger_path = os.path.join(opt.log_dir, opt.name)
#     if not os.path.exists(opt.log_dir):
#         os.mkdir(opt.log_dir)
#     if not os.path.exists(logger_path):
#         os.mkdir(logger_path)
#     logger = get_logger(logger_path, 'result')

#     #model = ourModel(opt)
#     model = ourModel2_Enhanced_BiLSTM(opt)

#     cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
#     best_model_name = f"best_model_{cur_time}.pth"

#     logger.info(f"splitwindow_time={configs['splitwindow_time']}, audiofeature_method={configs['audiofeature_method']}, "
#                 f"videofeature_method={configs['videofeature_method']}")
#     logger.info(f"batch_size={configs['batch_size']}, num_epochs={configs['num_epochs']}, "
#                 f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}, lr={opt.lr}")


#     seed = configs['seed']
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
    
#     # training
#     train_model(
#         train_json=configs['train_json'],
#         model=model,
#         max_len=opt.feature_max_len,
#         best_model_name=best_model_name,
#         audio_path=audio_path,
#         video_path=video_path,
#         seed=configs['seed']
#     )


from datetime import datetime
import os,json,time,argparse,torch,yaml
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from train_val_split import train_val_split1, train_val_split2
from models.our.our_model import *
from dataset import *
from utils.logger import get_logger
import numpy as np


def GetYAMLConfigs(path):
    with open(path, 'r') as file:
        c = yaml.safe_load(file)
    return c

class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def eval(model, val_loader, device):
    model.eval()
    total_emo_pred = []
    total_emo_label = []

    with torch.no_grad():
        for data in val_loader:
            for k, v in data.items():
                data[k] = v.to(device)
            model.set_input(data)
            model.test()
            emo_pred = model.emo_pred.argmax(dim=1).cpu().numpy()
            emo_label = data['emo_label'].cpu().numpy()
            total_emo_pred.append(emo_pred)
            total_emo_label.append(emo_label)

    total_emo_pred = np.concatenate(total_emo_pred)
    total_emo_label = np.concatenate(total_emo_label)

    # Classification metrics
    emo_acc_unweighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=None)
    class_counts = np.bincount(total_emo_label)
    sample_weights = 1 / (class_counts[total_emo_label] + 1e-6)
    emo_acc_weighted = accuracy_score(total_emo_label, total_emo_pred, sample_weight=sample_weights)
    emo_f1_weighted = f1_score(total_emo_label, total_emo_pred, average='weighted')
    emo_f1_unweighted = f1_score(total_emo_label, total_emo_pred, average='macro')
    emo_cm = confusion_matrix(total_emo_label, total_emo_pred)

    return total_emo_label, total_emo_pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm




def train_model(train_json, model, audio_path='', video_path='', max_len=5,
                best_model_name='best_model.pth', seed=None):

    logger.info(f"personalized features used: {configs['personalized_features_file']}")

    num_epochs = configs['num_epochs']
    device = configs['device']
    print(f"device: {device}")
    model.to(device)

    # split training and validation set
    if configs['track_option']=='Track1':
        train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=0.1, random_seed=seed)
    elif configs['track_option']=='Track2':
        train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=0.1, seed=seed)

    train_loader = DataLoader(
        AudioVisualDataset(train_data, configs['labelcount'], configs['personalized_features_file'], 
                          max_len=opt.feature_max_len, batch_size=configs['batch_size'],
                          audio_path=audio_path, video_path=video_path, 
                          use_personalized=configs.get('use_personalized', True)), 
        batch_size=configs['batch_size'], shuffle=True)

    val_loader = DataLoader(
        AudioVisualDataset(val_data, configs['labelcount'], configs['personalized_features_file'], 
                          max_len=opt.feature_max_len, batch_size=configs['batch_size'],
                          audio_path=audio_path, video_path=video_path, 
                          use_personalized=configs.get('use_personalized', True)), 
        batch_size=configs['batch_size'], shuffle=True)
    
    logger.info('The number of training samples = %d' % len(train_loader.dataset))
    logger.info('The number of val samples = %d' % len(val_loader.dataset))

    best_emo_acc = 0.0
    best_emo_f1 = 0.0
    best_emo_epoch = 1
    best_emo_cm = []

    for epoch in range(num_epochs):
        model.train(True)
        total_loss = 0
        total_ce_loss = 0
        total_focal_loss = 0
        
        # if epoch % 10 == 0:
        #     weights_info = model.get_current_weights()
        #     logger.info(f"Epoch {epoch}: CE weights: {weights_info['ce_weights']}")
        #     logger.info(f"Epoch {epoch}: Focal weights: {weights_info['focal_weights']}")
        #     logger.info(f"Epoch {epoch}: Task weights: {weights_info['task_weights']}")
            

        for i, data in enumerate(train_loader):
            for k, v in data.items():
            #     data[k] = v.to(device)
                if k in ['A_feat', 'V_feat', 'personalized_feat']:
                    data[k] = v.float().to(device)
                else:
                    data[k] = v.to(device)
                    
            # if i == 0:  # Print once per epoch
            #     print(f"Epoch {epoch}, Batch {i}:")
            #     print(f"Audio feat range: [{data['A_feat'].min():.4f}, {data['A_feat'].max():.4f}]")
            #     print(f"Visual feat range: [{data['V_feat'].min():.4f}, {data['V_feat'].max():.4f}]")
            #     print(f"Labels: {data['emo_label'].unique()}")
                
            model.set_input(data)
            model.optimize_parameters(epoch)

            # Get all losses from the TTFNet model
            losses = model.get_current_losses()
            total_loss += losses.get('total', 0)
            total_ce_loss += losses.get('emo_CE', 0)
            total_focal_loss += losses.get('emo_focal', 0)
            
            # if i % 50 == 0:
            #     print(f"Batch {i}: CE={losses.get('emo_CE', 0):.6f}, "
            #           f"Focal={losses.get('emo_focal', 0):.6f}, "
            #           f"Total={losses.get('total', 0):.6f}")
            
        
        # Calculate average losses
        avg_total_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_focal_loss = total_focal_loss / len(train_loader)

        # evaluation
        label, pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm = eval(model, val_loader, device)

        # Enhanced logging
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Losses - Total: {avg_total_loss:.6f}, CE: {avg_ce_loss:.6f}, Focal: {avg_focal_loss:.6f}")
        logger.info(f"Classification - Weighted F1: {emo_f1_weighted:.4f}, Unweighted F1: {emo_f1_unweighted:.4f}")
        logger.info(f"Classification - Weighted Acc: {emo_acc_weighted:.4f}, Unweighted Acc: {emo_acc_unweighted:.4f}")
        logger.info('Confusion Matrix:\n{}'.format(emo_cm))

        if emo_f1_weighted > best_emo_f1:
            cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
            best_emo_f1 = emo_f1_weighted
            best_emo_f1_unweighted = emo_f1_unweighted
            best_emo_acc = emo_acc_weighted
            best_emo_acc_unweighted = emo_acc_unweighted
            best_emo_cm = emo_cm
            best_emo_epoch = epoch + 1
            best_model = model
            save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name), best_model_name)
            torch.save(model.state_dict(), save_path)
            print("Saved best model.")

    logger.info(f"Training complete. Random seed: {seed}. Best epoch: {best_emo_epoch}.")
    logger.info(f"Best Classification - Weighted F1: {best_emo_f1:.4f}, Unweighted F1: {best_emo_f1_unweighted:.4f}")
    logger.info(f"Best Classification - Weighted Acc: {best_emo_acc:.4f}, Unweighted Acc: {best_emo_acc_unweighted:.4f}")
    logger.info('Best Confusion Matrix:\n{}'.format(best_emo_cm))

    # Update CSV output (removed regression metrics)
    csv_file = f'{opt.log_dir}/{opt.name}.csv'
    formatted_best_emo_cm = ' '.join([f"[{' '.join(map(str, row))}]" for row in best_emo_cm])
    header = f"Time,random seed,splitwindow_time,labelcount,audiofeature_method,videofeature_method," \
             f"batch_size,num_epochs,feature_max_len,lr," \
             f"Weighted_F1,Unweighted_F1,Weighted_Acc,Unweighted_Acc,Confusion_Matrix"
    result_value = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{seed},{configs['splitwindow_time']},{configs['labelcount']},{configs['audiofeature_method']},{configs['videofeature_method']}," \
               f"{configs['batch_size']},{configs['num_epochs']},{opt.feature_max_len},{opt.lr:.6f}," \
               f"{best_emo_f1:.4f},{best_emo_f1_unweighted:.4f},{best_emo_acc:.4f},{best_emo_acc_unweighted:.4f},{formatted_best_emo_cm}"

    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a') as file:
        if not file_exists:
            file.write(header + '\n')
        file.write(result_value + '\n')

    return best_emo_f1, best_emo_f1_unweighted, best_emo_acc, best_emo_acc_unweighted, best_emo_cm


# import torch.optim.lr_scheduler as lr_scheduler

# def train_model_with_lr_schedule(train_json, model, audio_path='', video_path='', max_len=5,
#                                 best_model_name='best_model.pth', seed=None):

#     logger.info(f"personalized features used: {configs['personalized_features_file']}")

#     num_epochs = configs['num_epochs']
#     device = configs['device']
#     print(f"device: {device}")
#     model.to(device)

#     # split training and validation set
#     if configs['track_option']=='Track1':
#         train_data, val_data, train_category_count, val_category_count = train_val_split1(train_json, val_ratio=0.1, random_seed=seed)
#     elif configs['track_option']=='Track2':
#         train_data, val_data, train_category_count, val_category_count = train_val_split2(train_json, val_percentage=0.1, seed=seed)

#     train_loader = DataLoader(
#         AudioVisualDataset(train_data, configs['labelcount'], configs['personalized_features_file'], 
#                           max_len=opt.feature_max_len, batch_size=configs['batch_size'],
#                           audio_path=audio_path, video_path=video_path, 
#                           use_personalized=configs.get('use_personalized', True)), 
#         batch_size=configs['batch_size'], shuffle=True)

#     val_loader = DataLoader(
#         AudioVisualDataset(val_data, configs['labelcount'], configs['personalized_features_file'], 
#                           max_len=opt.feature_max_len, batch_size=configs['batch_size'],
#                           audio_path=audio_path, video_path=video_path, 
#                           use_personalized=configs.get('use_personalized', True)), 
#         batch_size=configs['batch_size'], shuffle=True)
    
#     logger.info('The number of training samples = %d' % len(train_loader.dataset))
#     logger.info('The number of val samples = %d' % len(val_loader.dataset))

#     # ===================== LEARNING RATE SCHEDULERS =====================
    
#     # Option 1: Cosine Annealing with Warm Restarts (Recommended for your case)
#     scheduler_cosine = lr_scheduler.CosineAnnealingWarmRestarts(
#         model.optimizer, 
#         T_0=50,  # Number of epochs for the first restart
#         T_mult=2,  # Factor to increase T_0 after each restart
#         eta_min=1e-6  # Minimum learning rate
#     )
    
#     # Option 2: ReduceLROnPlateau (Reduces LR when validation loss plateaus)
#     scheduler_plateau = lr_scheduler.ReduceLROnPlateau(
#         model.optimizer,
#         mode='min',  # Looking for minimum validation loss
#         factor=0.5,  # Reduce LR by half
#         patience=10,  # Wait 10 epochs before reducing
#         min_lr=1e-6
#     )
    
#     # Option 3: Multi-Step LR (Reduces LR at specific epochs)
#     scheduler_multistep = lr_scheduler.MultiStepLR(
#         model.optimizer,
#         milestones=[100, 200, 350, 450],  # Epochs to reduce LR
#         gamma=0.5
#     )
    
#     # Option 4: Exponential Decay
#     scheduler_exp = lr_scheduler.ExponentialLR(
#         model.optimizer,
#         gamma=0.995
#     )
    
#     # Option 5: Custom Warm-up + Cosine Decay
#     def warm_up_cosine_lr_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
#         def lr_lambda(epoch):
#             if epoch < warmup_epochs:
#                 # Linear warm-up
#                 return (epoch + 1) / warmup_epochs
#             else:
#                 # Cosine decay
#                 progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
#                 return eta_min + (1 - eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
        
#         return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
#     scheduler_custom = warm_up_cosine_lr_scheduler(
#         model.optimizer, 
#         warmup_epochs=25, 
#         total_epochs=num_epochs,
#         eta_min=1e-6
#     )
    
#     # Choose which scheduler to use (recommended: scheduler_custom or scheduler_cosine)
#     scheduler = scheduler_custom  # You can change this to any of the above
#     scheduler_type = "custom_warmup_cosine"  # For logging
    
#     logger.info(f"Using {scheduler_type} learning rate scheduler")
#     logger.info(f"Initial learning rate: {model.optimizer.param_groups[0]['lr']}")

#     # ===================== TRAINING LOOP =====================
    
#     best_emo_acc = 0.0
#     best_emo_f1 = 0.0
#     best_emo_epoch = 1
#     best_emo_cm = []
#     best_val_loss = float('inf')
    
#     # Learning rate tracking
#     lr_history = []
#     val_loss_history = []

#     for epoch in range(num_epochs):
#         model.train(True)
#         total_loss = 0
#         total_ce_loss = 0
#         total_focal_loss = 0
        
#         # Track current learning rate
#         current_lr = model.optimizer.param_groups[0]['lr']
#         lr_history.append(current_lr)

#         for i, data in enumerate(train_loader):
#             for k, v in data.items():
#                 if k in ['A_feat', 'V_feat', 'personalized_feat']:
#                     data[k] = v.float().to(device)
#                 else:
#                     data[k] = v.to(device)
                    
#             model.set_input(data)
#             model.optimize_parameters(epoch)

#             # Get all losses
#             losses = model.get_current_losses()
#             total_loss += losses.get('emo_combined', 0)
#             total_ce_loss += losses.get('cb_loss', 0)
#             total_focal_loss += losses.get('focal_loss', 0)
        
#         # Calculate average losses
#         avg_total_loss = total_loss / len(train_loader)
#         avg_ce_loss = total_ce_loss / len(train_loader)
#         avg_focal_loss = total_focal_loss / len(train_loader)

#         # Evaluation
#         label, pred, emo_acc_weighted, emo_acc_unweighted, emo_f1_weighted, emo_f1_unweighted, emo_cm = eval(model, val_loader, device)
        
#         # Calculate validation loss for plateau scheduler
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for data in val_loader:
#                 for k, v in data.items():
#                     if k in ['A_feat', 'V_feat', 'personalized_feat']:
#                         data[k] = v.float().to(device)
#                     else:
#                         data[k] = v.to(device)
#                 model.set_input(data)
#                 model.forward()
#                 if hasattr(model, 'criterion_combined'):
#                     loss, _ = model.criterion_combined(model.emo_logits, model.emo_label)
#                     val_loss += loss.item()
        
#         avg_val_loss = val_loss / len(val_loader)
#         val_loss_history.append(avg_val_loss)

#         # ===================== LEARNING RATE SCHEDULING =====================
        
#         # Apply learning rate scheduling AFTER optimization step
#         if scheduler_type == "plateau":
#             scheduler.step(avg_val_loss)  # ReduceLROnPlateau needs validation loss
#         else:
#             scheduler.step()  # Other schedulers step based on epoch
        
#         # Enhanced logging with learning rate info
#         logger.info(f"Epoch {epoch + 1}/{num_epochs} | LR: {current_lr:.2e}")
#         logger.info(f"Losses - Total: {avg_total_loss:.6f}, CE: {avg_ce_loss:.6f}, Focal: {avg_focal_loss:.6f}, Val: {avg_val_loss:.6f}")
#         logger.info(f"Classification - Weighted F1: {emo_f1_weighted:.4f}, Unweighted F1: {emo_f1_unweighted:.4f}")
#         logger.info(f"Classification - Weighted Acc: {emo_acc_weighted:.4f}, Unweighted Acc: {emo_acc_unweighted:.4f}")
        
#         # Log learning rate changes
#         new_lr = model.optimizer.param_groups[0]['lr']
#         if abs(new_lr - current_lr) > 1e-8:
#             logger.info(f"Learning rate changed: {current_lr:.2e} -> {new_lr:.2e}")
        
#         logger.info('Confusion Matrix:\n{}'.format(emo_cm))

#         # Save best model
#         if emo_f1_weighted > best_emo_f1:
#             best_emo_f1 = emo_f1_weighted
#             best_emo_f1_unweighted = emo_f1_unweighted
#             best_emo_acc = emo_acc_weighted
#             best_emo_acc_unweighted = emo_acc_unweighted
#             best_emo_cm = emo_cm
#             best_emo_epoch = epoch + 1
#             best_val_loss = avg_val_loss
            
#             save_path = os.path.join(os.path.join(opt.checkpoints_dir, opt.name), best_model_name)
#             torch.save(model.state_dict(), save_path)
#             logger.info("Saved best model!")
        
#         # Early stopping check (optional)
#         if epoch > 100 and current_lr < 1e-7:
#             logger.info(f"Learning rate too small ({current_lr:.2e}), stopping early...")
#             break

#     # ===================== POST-TRAINING ANALYSIS =====================
    
#     logger.info(f"Training complete. Random seed: {seed}. Best epoch: {best_emo_epoch}.")
#     logger.info(f"Best Classification - Weighted F1: {best_emo_f1:.4f}, Unweighted F1: {best_emo_f1_unweighted:.4f}")
#     logger.info(f"Best Classification - Weighted Acc: {best_emo_acc:.4f}, Unweighted Acc: {best_emo_acc_unweighted:.4f}")
#     logger.info(f"Best Validation Loss: {best_val_loss:.6f}")
#     logger.info('Best Confusion Matrix:\n{}'.format(best_emo_cm))
    
#     # Save learning rate history
#     lr_log_path = os.path.join(opt.log_dir, f"{opt.name}_lr_history.txt")
#     with open(lr_log_path, 'w') as f:
#         f.write("Epoch,Learning_Rate,Val_Loss\n")
#         for i, (lr, val_loss) in enumerate(zip(lr_history, val_loss_history)):
#             f.write(f"{i+1},{lr:.8e},{val_loss:.6f}\n")
    
#     logger.info(f"Learning rate history saved to: {lr_log_path}")

#     # Update CSV output with LR info
#     csv_file = f'{opt.log_dir}/{opt.name}.csv'
#     formatted_best_emo_cm = ' '.join([f"[{' '.join(map(str, row))}]" for row in best_emo_cm])
#     header = f"Time,random seed,splitwindow_time,labelcount,audiofeature_method,videofeature_method," \
#              f"batch_size,num_epochs,feature_max_len,lr,scheduler_type," \
#              f"Weighted_F1,Unweighted_F1,Weighted_Acc,Unweighted_Acc,Best_Val_Loss,Confusion_Matrix"
#     result_value = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{seed},{configs['splitwindow_time']},{configs['labelcount']},{configs['audiofeature_method']},{configs['videofeature_method']}," \
#                f"{configs['batch_size']},{configs['num_epochs']},{opt.feature_max_len},{opt.lr:.6f},{scheduler_type}," \
#                f"{best_emo_f1:.4f},{best_emo_f1_unweighted:.4f},{best_emo_acc:.4f},{best_emo_acc_unweighted:.4f},{best_val_loss:.6f},{formatted_best_emo_cm}"

#     file_exists = os.path.exists(csv_file)
#     with open(csv_file, mode='a') as file:
#         if not file_exists:
#             file.write(header + '\n')
#         file.write(result_value + '\n')

#     return best_emo_f1, best_emo_f1_unweighted, best_emo_acc, best_emo_acc_unweighted, best_emo_cm


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument("--config", type=str, required=True)
    A = P.parse_args()
    configs = GetYAMLConfigs(path=A.config)

    config = load_config('config.json')
    opt = Opt(config)
    opt.use_personalized = configs['use_personalized']
    opt.emo_output_dim = configs['labelcount']
    opt.feature_max_len = configs['feature_max_len']
    opt.lr = float(configs['lr'])
    
    opt.warmup_epochs = configs.get('warmup_epochs', 25)
    opt.lr_policy = configs.get('lr_policy', 'step')
    opt.lr_decay_step = configs.get('lr_decay_step', 10)
    opt.lr_decay_gamma = configs.get('lr_decay_gamma', 0.1)
    
    opt.ce_weight = configs.get('ce_weight', 0.2)
    opt.focal_weight = configs.get('focal_weight', 0.4)
    opt.cb_weight = configs.get('cb_weight', 0.4)
    


    configs['data_rootpath'] = "/home/hiddenrock/DepressionDetection/MPDD-Young"
    
    audio_path = os.path.join(configs['data_rootpath'], 'Training', f"{configs['splitwindow_time']}", 'Audio', f"{configs['audiofeature_method']}") + '/'
    video_path = os.path.join(configs['data_rootpath'], 'Training', f"{configs['splitwindow_time']}", 'Visual', f"{configs['videofeature_method']}") + '/'

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Directory not found: {audio_path}")

    for filename in os.listdir(audio_path):
        if filename.endswith('.npy'):
            opt.input_dim_a = np.load(audio_path + filename).shape[1]
            break

    for filename in os.listdir(video_path):
        if filename.endswith('.npy'):
            opt.input_dim_v = np.load(video_path + filename).shape[1]            
            break

    opt.name = f"{configs['splitwindow_time']}_{configs['labelcount']}labels_{configs['audiofeature_method']}+{configs['videofeature_method']}"

    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    logger = get_logger(logger_path, 'result')

    # Use the TTFNet-inspired model (classification only)
    model =ourModel_Modified(opt)

    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
    best_model_name = f"best_model_{cur_time}.pth"

    logger.info(f"splitwindow_time={configs['splitwindow_time']}, audiofeature_method={configs['audiofeature_method']}, "
                f"videofeature_method={configs['videofeature_method']}")
    logger.info(f"batch_size={configs['batch_size']}, num_epochs={configs['num_epochs']}, "
                f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}, lr={opt.lr}")

    seed = configs['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_model(
        train_json=configs['train_json'],
        model=model,
        max_len=opt.feature_max_len,
        best_model_name=best_model_name,
        audio_path=audio_path,
        video_path=video_path,
        seed=configs['seed']
    )
    
