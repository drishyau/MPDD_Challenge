# import os
# import torch
# import json
# from models.our.our_model import ourModel, ourModel2_BiLSTM, Model5, AdvancedDepressionModel, 
# from train import eval
# import argparse
# from utils.logger import get_logger
# import numpy as np
# import pandas as pd
# import time,yaml
# from torch.utils.data import DataLoader
# from dataset import *

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

# if __name__ == '__main__':


#     P = argparse.ArgumentParser()
#     P.add_argument("--config", type=str, required=True)
#     A = P.parse_args()
#     configs = GetYAMLConfigs(path=A.config)
    
#     #args.test_json = os.path.join(args.data_rootpath, 'Testing', 'labels', 'Testing_files.json')
#     #args.personalized_features_file = os.path.join(args.data_rootpath, 'Testing', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')


#     config = load_config('config.json')
#     opt = Opt(config)
#     opt.use_personalized = configs['use_personalized']
#     opt.use_wav2vec = configs['use_wav2vec']

#     # Modify individual dynamic parameters in opt according to task category
#     opt.emo_output_dim = configs['labelcount']
#     opt.feature_max_len = configs['feature_max_len']
#     opt.lr = float(configs['lr'])

#     configs['data_rootpath'] = "/home/hiddenrock/DepressionDetection/TestData/MPDD-Elderly"
    
#     audio_path = os.path.join(configs['data_rootpath'],f"{configs['splitwindow_time']}", 'Audio', f"{configs['audiofeature_method']}") + '/'
#     video_path = os.path.join(configs['data_rootpath'],f"{configs['splitwindow_time']}", 'Visual', f"{configs['videofeature_method']}") + '/'

#     print(audio_path)
#     print(video_path)
    
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

#     opt.name = f'{configs['splitwindow_time']}_{configs['labelcount']}labels_{configs['audiofeature_method']}+{configs['videofeature_method']}'
#     logger_path = os.path.join(opt.log_dir, opt.name)
#     if not os.path.exists(opt.log_dir):
#         os.mkdir(opt.log_dir)
#     if not os.path.exists(logger_path):
#         os.mkdir(logger_path)
#     logger = get_logger(logger_path, 'result')

#     cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
#     best_model_name = f"best_model_{cur_time}.pth"

#     logger.info(f"splitwindow_time={configs['splitwindow_time']}, audiofeature_method={configs['audiofeature_method']}, "
#                 f"videofeature_method={configs['videofeature_method']}")
#     logger.info(f"batch_size={configs['batch_size']}, , "
#                 f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}")


#     model = AdvancedDepressionModel(opt)
#     model.load_state_dict(torch.load(configs['train_model']))
#     model.to(configs['device'])
#     test_data = json.load(open(configs['test_json'], 'r'))
#     test_loader = DataLoader(
#         AudioVisualDataset(test_data, configs['labelcount'], configs['personalized_features_file'], opt.feature_max_len,
#                            batch_size=configs['batch_size'],
#                            audio_path=audio_path, video_path=video_path, 
#                            isTest=True), batch_size=configs['batch_size'], shuffle=False)
#     logger.info('The number of testing samples = %d' % len(test_loader.dataset))

#     # testing
#     _, pred, *_ = eval(model, test_loader, configs['device'])

#     filenames = [item["audio_feature_path"] for item in test_data if "audio_feature_path" in item]
#     IDs = [path[:path.find('.')] for path in filenames]

#     if configs['labelcount']==2:
#         label="bin"
#     elif configs['labelcount']==3:
#         label="tri"
#     elif configs['labelcount']==5:
#         label="pen"
    

#     # output results to CSV
#     pred_col_name = f"{configs['splitwindow_time']}_{label}"

#     result_dir = f"/home/hiddenrock/DepressionDetection/answer_{configs['track_option']}"
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir)

#     csv_file = f"{result_dir}/submission.csv"

#     # Get the order of the IDs in the test data to ensure consistency
#     if configs['track_option']=='Track1':
#         test_ids = [item["audio_feature_path"].split('_')[0] + '_' + item["audio_feature_path"].split('_')[2] for item in test_data]
#     elif configs['track_option']=='Track2':
#         test_ids = ['_'.join([part.lstrip('0') for part in item["audio_feature_path"].replace(".npy", "").split('_')]) for item in test_data]

#     if os.path.exists(csv_file):
#         df = pd.read_csv(csv_file)
#     else:
#         df = pd.DataFrame(columns=["ID"])

#     if "ID" in df.columns:
#         df = df.set_index("ID")  
#     else:
#         df = pd.DataFrame(index=test_ids)

#     df.index.name = "ID"

#     pred = np.array(pred) 
#     if len(pred) != len(test_ids):
#         logger.error(f"Prediction length {len(pred)} does not match test ID length {len(test_ids)}")
#         raise ValueError("Mismatch between predictions and test IDs")

#     new_df = pd.DataFrame({pred_col_name: pred}, index=test_ids)
#     df[pred_col_name] = new_df[pred_col_name]
#     df = df.reindex(test_ids)
#     df.to_csv(csv_file)

#     logger.info(f"Testing complete. Results saved to: {csv_file}.")


import os
import torch
import json
from models.our.our_model import *
from train import eval
import argparse
from utils.logger import get_logger
import numpy as np
import pandas as pd
import time,yaml
from torch.utils.data import DataLoader
from dataset import *

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

if __name__ == '__main__':


    P = argparse.ArgumentParser()
    P.add_argument("--config", type=str, required=True)
    A = P.parse_args()
    configs = GetYAMLConfigs(path=A.config)
    
    #args.test_json = os.path.join(args.data_rootpath, 'Testing', 'labels', 'Testing_files.json')
    #args.personalized_features_file = os.path.join(args.data_rootpath, 'Testing', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')


    config = load_config('config.json')
    opt = Opt(config)
    opt.use_personalized = configs['use_personalized']

    # Modify individual dynamic parameters in opt according to task category
    opt.emo_output_dim = configs['labelcount']
    opt.feature_max_len = configs['feature_max_len']
    opt.lr = float(configs['lr'])

    configs['data_rootpath'] = "/home/hiddenrock/DepressionDetection/TestData/MPDD-Young"
    
    audio_path = os.path.join(configs['data_rootpath'],f"{configs['splitwindow_time']}", 'Audio', f"{configs['audiofeature_method']}") + '/'
    video_path = os.path.join(configs['data_rootpath'],f"{configs['splitwindow_time']}", 'Visual', f"{configs['videofeature_method']}") + '/'

    
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

    opt.name = f'{configs['splitwindow_time']}_{configs['labelcount']}labels_{configs['audiofeature_method']}+{configs['videofeature_method']}'
    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    logger = get_logger(logger_path, 'result')

    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
    best_model_name = f"best_model_{cur_time}.pth"

    logger.info(f"splitwindow_time={configs['splitwindow_time']}, audiofeature_method={configs['audiofeature_method']}, "
                f"videofeature_method={configs['videofeature_method']}")
    logger.info(f"batch_size={configs['batch_size']}, , "
                f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}")


    model = Model23(opt)
    model.load_state_dict(torch.load(configs['train_model']))
    model.to(configs['device'])
    test_data = json.load(open(configs['test_json'], 'r'))
    test_loader = DataLoader(
        AudioVisualDataset(test_data, configs['labelcount'], configs['personalized_features_file'], opt.feature_max_len,
                           batch_size=configs['batch_size'],
                           audio_path=audio_path, video_path=video_path, 
                           isTest=True), batch_size=configs['batch_size'], shuffle=False)
    logger.info('The number of testing samples = %d' % len(test_loader.dataset))

    # # testing
    _, pred, *_ = eval(model, test_loader, configs['device'])

    filenames = [item["audio_feature_path"] for item in test_data if "audio_feature_path" in item]
    IDs = [path[:path.find('.')] for path in filenames]

    print(filenames)
    print(IDs)
    if configs['labelcount']==2:
        label="bin"
    elif configs['labelcount']==3:
        label="tri"
    elif configs['labelcount']==5:
        label="pen"
    

    # output results to CSV
    pred_col_name = f"{configs['splitwindow_time']}_{label}"

    result_dir = f"/home/hiddenrock/DepressionDetection/answer_{configs['track_option']}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    csv_file = f"{result_dir}/submission.csv"

    # Get the order of the IDs in the test data to ensure consistency
    if configs['track_option']=='Track1':
        test_ids = [item["audio_feature_path"].replace(".npy", "") for item in test_data]
    elif configs['track_option']=='Track2':
        test_ids = ['_'.join([part.lstrip('0') for part in item["audio_feature_path"].replace(".npy", "").split('_')]) for item in test_data]

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["ID"])

    if "ID" in df.columns:
        df = df.set_index("ID")  
    else:
        df = pd.DataFrame(index=test_ids)

    df.index.name = "ID"

    pred = np.array(pred) 
    if len(pred) != len(test_ids):
        logger.error(f"Prediction length {len(pred)} does not match test ID length {len(test_ids)}")
        raise ValueError("Mismatch between predictions and test IDs")

    
    new_df = pd.DataFrame({pred_col_name: pred}, index=test_ids)
    df[pred_col_name] = new_df[pred_col_name]
    df = df.reindex(test_ids)
    df.to_csv(csv_file)

    logger.info(f"Testing complete. Results saved to: {csv_file}.")
