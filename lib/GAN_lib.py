import os
import sys

import torch
import numpy as np
import random

GLOBAL_SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # si tienes varias GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")
set_seed(GLOBAL_SEED)


import matplotlib.pyplot as plt

sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../deep4downscaling/')
import deep.loss as deep_loss
import deep.train as deep_train
import deep.models as deep_models
import deep.pred as deep_pred
import deep.utils as deep_utils


sys.path.append('../lib/')
import ANA_lib
import aux_lib
import derived_predictors
import DL_lib
import GAN_lib
import down_scene_ANA
import down_scene_DL
import down_scene_GAN
import down_scene_MOS
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_methods
import grids
import launch_jobs
import launch_jobs_GPU
import MOS_lib
import plot
import postpro_lib
import postprocess
import precontrol
import preprocess
import process
import read
import transform
import TF_lib
import val_lib
import WG_lib
import write

########################################################################################################################
def train(targetVar, methodName, family, mode, fields):
    '''
    Calibrates regression for all points,divided in chunks if run at HPC.
    '''

    lambda_adv = gan_lambda_adv
    lambda_recon = gan_lambda_recon
    freq_train_gen = gan_freq_train_gen
    freq_train_disc = gan_freq_train_disc


    device = ('cuda' if torch.cuda.is_available() else 'cpu')


    # Define pathOut
    pathOut = pathAux + 'TRAINED_MODELS/' + targetVar.upper() + '/' + methodName + '/'

    # Declares variables for father process, who creates pathOut
    try:
        os.makedirs(pathOut)
    except:
        pass

    # Load data (X_train)
    X_train = np.load(pathAux+'TRANSFORMATION/SPRED/'+targetVar+'_training.npy')
    X_train = X_train.astype('float32')

    # Load data (y_train)
    y_train = read.hres_data(targetVar, period='training')['data']
    y_train = (100 * y_train).astype(predictands_codification[targetVar]['type'])
    special_value = int(100 * predictands_codification[targetVar]['special_value'])
    y_train = (y_train).astype('float')
    y_train[y_train >= special_value] = np.nan
    y_train /= 100
    y_train = y_train.astype('float32')

    # # y_mean, y_std = np.nanmean(y_train, axis=0), np.nanstd(y_train, axis=0),
    # y_mean, y_std = np.nanmean(y_train), np.nanstd(y_train),
    # # y_train = (y_train - y_mean) / y_std
    # os.makedirs(pathAux+'TRANSFORMATION/Y/', exist_ok=True)
    # np.save(pathAux+'TRANSFORMATION/Y/'+targetVar+'_mean.npy', y_mean)
    # np.save(pathAux+'TRANSFORMATION/Y/'+targetVar+'_std.npy', y_std)

    # Remove days with Nans
    invalid_y = list(set(np.where(np.isnan(y_train))[0]))
    invalid_X = list(set(np.where(np.isnan(X_train))[0]))
    valid = [i for i in range(y_train.shape[0]) if (i not in invalid_y) and (i not in invalid_X)]
    X_train = X_train[valid]
    y_train = y_train[valid]
    training_dates_valid = [training_dates[i] for i in valid]

    if targetVar in asym_loss_parameters:
        asym_path = pathAux + 'ASYM/' + targetVar + '/'
        os.makedirs(asym_path, exist_ok=True)
        loss_function = deep_loss.Asym(ignore_nans=True, asym_path=asym_path,
                                       asym_weight=asym_loss_parameters[targetVar]['asym_weight'],
                                       cdf_pow=asym_loss_parameters[targetVar]['cdf_pow'],
                                       r01_asym_weight=asym_loss_parameters[targetVar]['r01_asym_weight'],
                                       r01_cdf_pow=asym_loss_parameters[targetVar]['r01_cdf_pow'],
                                       )
        if loss_function.parameters_exist():
            loss_function.load_parameters()
        else:
            y_train_ds = xr.Dataset(
                {targetVar: (["time", "point"], y_train)},
                coords={"time": training_dates_valid, "point": range(y_train.shape[1])}
            )
            loss_function.compute_parameters(data=y_train_ds, var_target=targetVar)

        # Either for new calculations or for loading parameters
        loss_function.prepare_parameters(device=device)
    elif targetVar in mseExtremes_loss_parameters:
        asym_path = pathAux + 'ASYM/' + targetVar + '/'
        os.makedirs(asym_path, exist_ok=True)
        loss_function = deep_loss.MseExtremesLoss(ignore_nans=True, asym_path=asym_path,
                                                  w=mseExtremes_loss_parameters[targetVar]['w'],
                                                  pow=mseExtremes_loss_parameters[targetVar]['pow'],
                                                  )
        if loss_function.parameters_exist():
            loss_function.load_parameters()
        else:
            y_train_ds = xr.Dataset(
                {targetVar: (["time", "point"], y_train)},
                coords={"time": training_dates_valid, "point": range(y_train.shape[1])}
            )
            loss_function.compute_parameters(data=y_train_ds, var_target=targetVar)

        # Either for new calculations or for loading parameters
        loss_function.prepare_parameters(device=device)
    else:
        loss_function = deep_loss.MseLoss(ignore_nans=True)


    # Create Dataset
    train_dataset = deep_utils.StandardDataset(x=X_train, y=y_train)

    # Split into training and validation sets
    train_size = int(0.9 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size], generator=g)

    # Create DataLoaders
    batch_size = 64

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  worker_init_fn=seed_worker, generator=g)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  worker_init_fn=seed_worker, generator=g)

    model_name = methodName+'-'+targetVar
    model_name_D = methodName+'-'+targetVar+'-D'

    if targetVar == 'pr':
        model_G = deep_models.DeepESDpr(x_shape=X_train.shape, y_shape=y_train.shape, filters_last_conv=1,
                                        stochastic=False)
    else:
        model_G = deep_models.DeepESDtas(x_shape=X_train.shape, y_shape=y_train.shape, filters_last_conv=1,
                                   stochastic=False)
    model_D = deep_models.DeepESD_Discriminator(x_shape=X_train.shape, y_shape=y_train.shape, filters_last_conv=1,)


    num_epochs = 400

    learning_rate = 0.0001
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learning_rate)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate)


    loop_return_dict = deep_train.standard_cgan_training_loop(targetVar, generator=model_G, discriminator=model_D,
                          gen_name=model_name, disc_name=model_name_D,
                          model_path=pathOut,
                          device=device, num_epochs=num_epochs,
                          loss_function=loss_function, optimizer_G=optimizer_G, optimizer_D=optimizer_D,
                          train_data=train_dataloader, valid_data=valid_dataloader,
                          lambda_adv=lambda_adv, lambda_recon=lambda_recon,
                          freq_train_gen=freq_train_gen, freq_train_disc=freq_train_disc,
                          save_checkpoint_every=50, resume_checkpoint=None, save_fake_every=10, fixed_indices=[i for i in range(20)])



########################################################################################################################

if __name__=="__main__":

    targetVar = sys.argv[1]
    methodName = sys.argv[2]
    family = sys.argv[3]
    mode = sys.argv[4]
    fields = sys.argv[5]

    train(targetVar, methodName, family, mode, fields)