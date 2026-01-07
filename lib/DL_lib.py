import sys

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

    # Adapt shapes for UNET
    if methodName == 'UNET':
        (X_train, y_train,
         hX_orig, wX_orig, nPointsY_orig,
         hX_adj, wX_adj, nPointsY_adj) = aux_lib.adjust_dimensions_for_UNET(X_train, Y=y_train,
                                                       hX_orig=X_train.shape[2], wX_orig=X_train.shape[3],
                                                       nPointsY_orig=y_train.shape[-1],)

    # Remove days with Nans
    invalid_y = list(set(np.where(np.isnan(y_train))[0]))
    invalid_X = list(set(np.where(np.isnan(X_train))[0]))
    valid = [i for i in range(y_train.shape[0]) if (i not in invalid_y) and (i not in invalid_X)]
    X_train = X_train[valid]
    y_train = y_train[valid]
    training_dates_valid = [training_dates[i] for i in valid]


    if targetVar in asym_loss_parameters:
        asym_path = pathAux + 'ASYM/'
        os.makedirs(asym_path, exist_ok=True)
        loss_function = deep_loss.Asym(ignore_nans=True, asym_path=asym_path,
                                       asym_weight=asym_loss_parameters[targetVar]['asym_weight'],
                                       cdf_pow=asym_loss_parameters[targetVar]['cdf_pow'])
        # if loss_function.parameters_exist():
        #     loss_function.load_parameters()
        # else:
        y_train_ds = xr.Dataset(
            {targetVar: (["time", "point"], y_train)},
            coords={"time": training_dates_valid, "point": range(y_train.shape[1])}
        )
        loss_function.compute_parameters(data=y_train_ds, var_target=targetVar)
        loss_function.prepare_parameters(device=device)
    else:
        loss_function = deep_loss.MseLoss(ignore_nans=True)

    # Create Dataset
    train_dataset = deep_utils.StandardDataset(x=X_train, y=y_train)

    # Split into training and validation sets
    train_size = int(0.9 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])


    # Create DataLoaders
    batch_size = 64

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=True)


    model_name = methodName+'-'+targetVar

    if targetVar == 'pr':
        if methodName == 'DeepESD':
            model = deep_models.DeepESDpr(x_shape=X_train.shape, y_shape=y_train.shape, filters_last_conv=1, stochastic=False)
        elif methodName == 'UNET':
            model = deep_models.UnetPr(x_shape=X_train.shape, y_shape=y_train.shape, stochastic=False,
                                       input_padding=(0, 0, 0, 0), kernel_size=3, padding="same",
                                       batch_norm=True, trans_conv=False)
    else:
        if methodName == 'DeepESD':
            model = deep_models.DeepESDtas(x_shape=X_train.shape, y_shape=y_train.shape, filters_last_conv=1,
                                       stochastic=False)
        elif methodName == 'UNET':
            model = deep_models.UnetTas(x_shape=X_train.shape, y_shape=y_train.shape, stochastic=False,
                                       input_padding=(0, 0, 0, 0), kernel_size=3, padding="same",
                                       batch_norm=True, trans_conv=False)


    num_epochs = 10000
    patience_early_stopping = 20

    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, val_loss = deep_train.standard_training_loop(
                                model=model, model_name=model_name, model_path=pathOut,
                                device=device, num_epochs=num_epochs,
                                loss_function=loss_function, optimizer=optimizer,
                                train_data=train_dataloader, valid_data=valid_dataloader,
                                patience_early_stopping=patience_early_stopping)


########################################################################################################################

if __name__=="__main__":

    targetVar = sys.argv[1]
    methodName = sys.argv[2]
    family = sys.argv[3]
    mode = sys.argv[4]
    fields = sys.argv[5]

    train(targetVar, methodName, family, mode, fields)