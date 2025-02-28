"""
This module contains the functions for training deep learning models.

Author: Jose González-Abad
"""

import os
import copy
import numpy as np
import time
import math
import torch

def standard_training_loop(model: torch.nn.Module, model_name: str, model_path: str,
                           loss_function: torch.nn.Module, optimizer: torch.optim,
                           num_epochs: int, device: str,
                           train_data: torch.utils.data.dataloader.DataLoader,
                           valid_data: torch.utils.data.dataloader.DataLoader=None,
                           scheduler: torch.optim=None,
                           patience_early_stopping: int=None,
                           mixed_precision: bool=False) -> dict:
    
    """
    Standard training loop for a DL model in a supervised setting. Besides the
    training, it is possible to perform a validation step and control the saving 
    of the model through an early stopping strategy. To activate the latter, pass
    a value to the argument patience_early_stopping, otherwise the training will 
    continue for the num_epochs specified, saving the model at the end of each
    epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model to train

    model_name : str
        Name of the model when saved as
        a .pt file

    model_path : str
        Path of the folder where the model
        will be saved

    loss_function : torch.nn.Module
        Loss function to use when training/evaluating
        the model

    optimizer : torch.optim
        Optimizer to use when training the model

    num_epochs : int
        Number of epochs

    device : str
        Device used to run the training (cuda or cpu)

    train_data : torch.utils.data.dataloader.DataLoader
        DataLoader with the training data

    valid_data : torch.utils.data.dataloader.DataLoader, optional
        DataLoader with the validation data

    scheduler : torch.optim=None, optional
        Scheduler to use for the optimization

    patience_early_stopping : int, optional
        Number of steps allowed for the model to run before
        any improvement in the loss function occurs. If this
        number is surpassd without improvement the training is
        stopped.

    mixed_precision : bool, optional
        If training on GPUs, mixed_precision allows for automatic
        mixed precision training to reduce computation and memory
        footprint. By default this parameter is set to False.

    Returns
    -------
    dict
        Dictionary with list(s) representing the loss function
        across epochs.
    """

    model = model.to(device)

    # The scaler scales the loss to avoid the underflow
    # of gradients
    if mixed_precision:
        scaler = torch.amp.GradScaler()

    # Set the early stopping parameters
    if patience_early_stopping is not None:
        best_val_loss = math.inf
        early_stopping_step = 0

    # Register the losses per epoch
    epoch_train_loss = []
    epoch_valid_loss = []

    # Iterate over epochs
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_train_loss.append(0)

        # Iterate over batches
        model.train()
        for x, y in train_data:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if mixed_precision: 
                with torch.amp.autocast(device_type=device):
                    output = model(x)
                    loss = loss_function(target=y, output=output)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(x)
                loss = loss_function(target=y, output=output)
                loss.backward()               
                optimizer.step()

            epoch_train_loss[-1] += loss.item()

        # Compute mean loss across the epoch
        epoch_train_loss[-1] = epoch_train_loss[-1] / len(train_data)

        # If valid data is provided, perform a pass through it
        if valid_data is not None:

            epoch_valid_loss.append(0)

            model.eval()
            for x, y in valid_data:

                x = x.to(device)
                y = y.to(device)    

                if mixed_precision: 
                    with torch.amp.autocast(device_type=device):
                        output = model(x)
                        loss = loss_function(target=y, output=output)
                else:
                        output = model(x)
                        loss = loss_function(target=y, output=output)                    

                epoch_valid_loss[-1] += loss.item()

            # Compute mean loss across the epoch
            epoch_valid_loss[-1] = epoch_valid_loss[-1] / len(valid_data)

        epoch_end = time.time()
        epoch_time = np.round(epoch_end - epoch_start, 2)

        log_msg = f'Epoch {epoch+1} ({epoch_time} secs) | Training Loss {np.round(epoch_train_loss[-1], 4)}'
        if valid_data is not None: log_msg = log_msg + f' Valid Loss {np.round(epoch_valid_loss[-1], 4)}'

        # If early stopping perform the step
        if patience_early_stopping is not None:

            # Save the model if the validation loss improves
            if epoch_valid_loss[-1] < best_val_loss:
                best_val_loss = epoch_valid_loss[-1]
                early_stopping_step = 0
                log_msg = log_msg + ' (Model saved)'
                torch.save(model.state_dict(),
                           os.path.expanduser(f'{model_path}/{model_name}.pt'))
            else:
                early_stopping_step +=1

            # If no improvement over the specified steps, stop the training
            if early_stopping_step >= patience_early_stopping:
                print(log_msg)
                print('***Training finished***')
                break

        else: # If early stopping is not configured save the model at each epoch
            log_msg = log_msg + ' (Model saved)'
            torch.save(model.state_dict(),
                       os.path.expanduser(f'{model_path}/{model_name}.pt'))
        
        # Print log
        print(log_msg)

    # Return loss functions
    if valid_data is not None:
        return epoch_train_loss, epoch_valid_loss
    else:
        return epoch_train_loss, None