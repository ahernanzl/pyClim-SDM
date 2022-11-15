import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import derived_predictors
import down_scene_ANA
import down_scene_MOS
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_methods
import grids
import launch_jobs
import MOS_lib
import plot
import postpro_lib
import postprocess
import precontrol
import preprocess
import process
import read
import standardization
import TF_lib
import val_lib
import WG_lib
import write

########################################################################################################################
def train_chunk(targetVar, methodName, family, mode, fields, iproc=0, nproc=1):
    '''
    Calibrates regression for all points,divided in chunks if run at HPC.
    '''

    # Define pathOut
    pathOut = pathAux + 'TRAINED_MODELS/' + targetVar.upper() + '/' + methodName + '/'

    # Declares variables for father process, who creates pathOut
    if iproc == 0 or running_at_HPC == False:

        try:
            os.makedirs(pathOut)
        except:
            pass

        # Load data (X_train)
        if 'pred' in fields:
            pred_calib = np.load(pathAux+'STANDARDIZATION/PRED/'+targetVar+'_training.npy')
            pred_calib = pred_calib.astype('float32')
            X_train = pred_calib
        if 'saf' in fields:
            saf_calib = np.load(pathAux+'STANDARDIZATION/SAF/'+targetVar+'_training.npy')
            saf_calib = saf_calib.astype('float32')
            X_train = saf_calib
        if 'var' in fields:
            var_calib = np.load(pathAux+'STANDARDIZATION/VAR/'+targetVar+'_training.npy')
            if 'pred' not in fields:
                X_train = var_calib
            else:
                # For Radom Forest and Extreme Gradient Boost mixing pred (standardized) and var (pcp) is allowed
                X_train = np.concatenate((X_train, var_calib), axis=1)

        # Load data (y_train and association)
        y_train = read.hres_data(targetVar, period='training')['data']
        y_train = (100 * y_train).astype(predictands_codification[targetVar]['type'])
        i_4nn = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper() + '_' + interp_mode + '/i_4nn.npy')
        j_4nn = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper() + '_' + interp_mode + '/j_4nn.npy')
        w_4nn = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper() + '_' + interp_mode + '/w_4nn.npy')

    # Declares variables for the other processes
    else:
        X_train = None
        y_train = None
        i_4nn = None
        j_4nn = None
        w_4nn = None

    # Share data with all subprocesses
    if nproc > 1 and running_at_HPC == True:
        X_train = MPI.COMM_WORLD.bcast(X_train, root=0)
        y_train = MPI.COMM_WORLD.bcast(y_train, root=0)
        i_4nn = MPI.COMM_WORLD.bcast(i_4nn, root=0)
        j_4nn = MPI.COMM_WORLD.bcast(j_4nn, root=0)
        w_4nn = MPI.COMM_WORLD.bcast(w_4nn, root=0)


    # create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(hres_npoints[targetVar]) / n_chunks))
    points_chunk = []
    for ichunk in range(n_chunks):
        points_chunk.append(list(range(hres_npoints[targetVar]))[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    ichunk = iproc
    npoints_ichunk = len(points_chunk[ichunk])

    # If we are tuning hyperparameters only certaing points will be calculated
    if plot_hyperparameters_epochs_nEstimators_featureImportances == True:
        points_chunk[ichunk]=[x for x in points_chunk[ichunk] if x%500==0]

    # loop through all points of the chunk
    special_value = int(100 * predictands_codification[targetVar]['special_value'])
    for ipoint in points_chunk[ichunk]:
        ipoint_local_index = points_chunk[ichunk].index(ipoint)
        if ipoint_local_index % 1 == 0:
            print('--------------------')
            print('ichunk:	', ichunk, '/', n_chunks)
            print('training', targetVar, methodName, round(100*ipoint_local_index/npoints_ichunk, 2), '%')

        # Get preds from neighbour/s and trains model for each point
        y_train_ipoint = y_train[:, ipoint]
        valid_y = np.where(y_train_ipoint < special_value)[0]
        invalid_X = list(set(np.where(np.isnan(X_train))[0]))
        valid = [x for x in valid_y if x not in invalid_X]
        if len(valid) < 30:
            exit('Not enough valid predictands to train')
        y_train_ipoint = y_train_ipoint[valid]
        X_train_ipoint = X_train[valid, :, :, :]

        # Prepare X_train shape
        if methodName not in methods_using_preds_from_whole_grid:
            X_train_ipoint = grids.interpolate_predictors(X_train_ipoint, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_mode)
        elif methodName not in ['CNN', ]:
            X_train_ipoint = X_train_ipoint.reshape(X_train_ipoint.shape[0], X_train_ipoint.shape[1], -1)

        # Train TF (clf and reg)
        reg, clf = train_point(targetVar, methodName, X_train_ipoint, y_train_ipoint, ipoint)

        # Save clf/reg. some objects can be serialized with pickle, but keras models have their own method
        try:
            pickle.dump(reg, open(pathOut + 'reg_'+str(ipoint), 'wb'))
        except:
            reg.save(pathOut + 'reg_'+str(ipoint) + '.h5')
        if targetVar == 'pr':
            try:
                pickle.dump(clf, open(pathOut + 'clf_'+str(ipoint), 'wb'))
            except:
                clf.save(pathOut + 'clf_'+str(ipoint) + '.h5')


########################################################################################################################
def train_point(targetVar, methodName, X, y, ipoint):
    '''
    Train model (classifiers and regressors)
    '''

    classifier = None
    regressor = None
    history_clf = None
    history_reg = None



    # Define callbacks for neural networks training
    tf_nn_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=2,
            verbose=0,
            mode='auto'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.25,
            patience=3,
        )
    ]


    # For precipitation trains classier+regression, but for temperature only regression
    if targetVar != 'pr':

        # Regressor
        if methodName == 'MLR':
            regressor = RidgeCV(cv=3)
            regressor.fit(X, y)

        elif methodName == 'SVM':
            # Use a random sample of 5000 data
            nDays = X.shape[0]
            nRand = min(nDays, 5000)
            rng = np.random.default_rng()
            iDays = rng.choice(nDays, size=nRand, replace=False)
            X, y = X[iDays], y[iDays]
            regressor = GridSearchCV(svm.SVR(kernel='rbf'),
                                param_grid={"C": np.logspace(3, 5, 3), "gamma": np.logspace(-3, 0, 3)}, cv=3)
            regressor.fit(X, y)

        elif methodName == 'LS-SVM':
            # Use a random sample of 5000 data
            nDays = X.shape[0]
            nRand = min(nDays, 5000)
            rng = np.random.default_rng()
            iDays = rng.choice(nDays, size=nRand, replace=False)
            X, y = X[iDays], y[iDays]
            regressor = GridSearchCV(KernelRidge(kernel='rbf'),
                                 param_grid={"alpha": np.logspace(-3, 0, 4), "gamma": np.logspace(-2, 2, 5)}, cv=3)
            regressor.fit(X, y)

        elif methodName == 'RF':
            regressor = GridSearchCV(RandomForestRegressor(), param_grid={"max_depth": [20, 60]}, cv=3)
            regressor.fit(X, y)

        elif methodName == 'XGB':
            regressor = XGBRegressor(n_estimators=100, max_depth=6, early_stopping_rounds=20, learning_rate=0.1)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
            regressor.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=0)
            history_reg = regressor.evals_result()

        elif methodName == 'ANN':
            # skelear implementation
            # regressor = MLPRegressor(hidden_layer_sizes=(200,), max_iter=100000)
            # regressor = GridSearchCV(MLPRegressor(max_iter=100000),
            #                          param_grid={'hidden_layer_sizes': [5,10,20,50,100,200]}, cv=3)
            # regressor.fit(X, y)

            # keras tensorflow implementation
            regressor = tf.keras.models.Sequential([
                keras.layers.Dense(8, activation='relu', input_shape=X.shape[1:]),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(1), ])
            regressor.compile(optimizer='adam', loss='mse', metrics=['mse'])
            history_reg = regressor.fit(X, y, epochs=100, validation_split=.2, verbose=0, callbacks=tf_nn_callbacks)

        elif methodName in ('CNN', ):
            # Prepare shape for convolution layer
            X = np.swapaxes(np.swapaxes(X, 1, 2), 2, 3)
            regressor = tf.keras.models.Sequential()
            if X.shape[1] < 5 or X.shape[2] < 5:
                regressor.add(layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu', input_shape=X.shape[1:]))
            else:
                regressor.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
            # if X.shape[1] >= 10 and X.shape[2] >= 10:
            #     regressor.add(layers.MaxPooling2D(pool_size=2))
            #     regressor.add(layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
            regressor.add(layers.Flatten())
            regressor.add(layers.Dense(16, activation='relu'))
            regressor.add(layers.Dense(8, activation='relu'))
            regressor.add(layers.Dense(1))
            regressor.compile(optimizer='adam', loss='mse', metrics=['mse'])
            history_reg = regressor.fit(X, y, epochs=100, validation_split=.2, verbose=0, callbacks=tf_nn_callbacks)

    else:
        # Prepare data for precipitation
        israiny = (y > (100 * wetDry_th))
        nDays = y.size
        # If all data are dry, clasiffiers won't work. A random data is forced into wet.
        minWetDays = 3
        allDry = False
        if len(np.where(israiny==False)[0]) == nDays:
            allDry = True
        if allDry == True:
            rand = random.sample(range(nDays), minWetDays)
            israiny[rand] = True
            y[rand] = 100*0.1
        X_rainy_days = X[israiny==True]
        if allDry == True:
            for i in range(minWetDays):
                X_rainy_days[i] = 999 - i
        y_rainy_days = y[israiny==True]

        # Classifier pcp
        if methodName[:3] == 'GLM':
            classifier = LogisticRegressionCV(cv=3, max_iter=1000)
            classifier = CalibratedClassifierCV(classifier, cv=5)
            classifier.fit(X, 1*israiny)

        elif methodName == 'SVM':
            classifier = GridSearchCV(svm.SVC(kernel='rbf'),
                                    param_grid={"C": np.logspace(0, 1, 2), "gamma": np.logspace(-2, -1, 2)}, cv=3)
            classifier = CalibratedClassifierCV(classifier, cv=5)
            classifier.fit(X, 1*israiny)

        elif methodName == 'LS-SVM':
            classifier = RidgeClassifierCV(cv=3)
            classifier = CalibratedClassifierCV(classifier, cv=5)
            classifier.fit(X, 1*israiny)

        elif methodName == 'RF':
            classifier = GridSearchCV(RandomForestClassifier(), param_grid={"max_depth": [20, 60]}, cv=3)
            if plot_hyperparameters_epochs_nEstimators_featureImportances == False:
                classifier = CalibratedClassifierCV(classifier, cv=5)
            classifier.fit(X, 1*israiny)

        elif methodName == 'XGB':
            classifier = XGBClassifier(n_estimators=100, max_depth=6, early_stopping_rounds=20, learning_rate=0.1)
            X_train, X_valid, y_train, y_valid = train_test_split(X, israiny, test_size=0.2)
            classifier.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=0)
            history_clf = classifier.evals_result()

        elif methodName == 'ANN':
            # sklearn implementation
            # classifier = MLPClassifier(hidden_layer_sizes=(200,), max_iter=100000)
            # classifier = GridSearchCV(MLPClassifier(max_iter=100000),
            #                          param_grid={'hidden_layer_sizes': [5,10,20,50,100,200]}, cv=3)
            # classifier.fit(X, 1*israiny)

            # keras tensorflow implementation
            classifier = tf.keras.models.Sequential([
                keras.layers.Dense(8, activation='relu', input_shape=X.shape[1:]),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid'), ])
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            history_clf = classifier.fit(X, 1*israiny, epochs=100, validation_split=.2, verbose=0, callbacks=tf_nn_callbacks)

        elif methodName in ['CNN', ]:
            # Prepare shape for convolution
            X = np.swapaxes(np.swapaxes(X, 1, 2), 2, 3)
            nfilters = 32
            classifier = tf.keras.models.Sequential()
            if X.shape[1] < 5 or X.shape[2] < 5:
                classifier.add(layers.Conv2D(filters=nfilters, kernel_size=(1, 1), activation='relu', input_shape=X.shape[1:]))
            else:
                classifier.add(layers.Conv2D(filters=nfilters, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
            # if X.shape[1] >= 10 and X.shape[2] >= 10:
            #     classifier.add(layers.MaxPooling2D(pool_size=2))
            #     classifier.add(layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
            classifier.add(layers.Flatten())
            classifier.add(layers.Dense(nfilters, activation='relu'))
            classifier.add(layers.Dense(8, activation='relu'))
            classifier.add(layers.Dense(1, activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            history_clf = classifier.fit(X, 1*israiny, epochs=100, validation_split=.2, verbose=0, callbacks=tf_nn_callbacks)

        # Regressor
        if methodName == 'GLM-LIN':
            regressor = RidgeCV(cv=3)
            regressor.fit(X_rainy_days, y_rainy_days)

        elif methodName == 'GLM-EXP':
            y_rainy_days = np.log(y_rainy_days)
            regressor = RidgeCV(cv=3)
            regressor.fit(X_rainy_days, y_rainy_days)

        elif methodName == 'GLM-CUB':
            y_rainy_days = np.cbrt(y_rainy_days)
            regressor = RidgeCV(cv=3)
            regressor.fit(X_rainy_days, y_rainy_days)

        elif methodName == 'SVM':
            regressor = GridSearchCV(svm.SVR(kernel='rbf'),
                                param_grid={"C": np.logspace(3, 5, 3), "gamma": np.logspace(-2, 0, 3)}, cv=3)
            regressor.fit(X_rainy_days, y_rainy_days)

        elif methodName == 'LS-SVM':
            regressor = GridSearchCV(KernelRidge(kernel='rbf'),
                                 param_grid={"alpha": np.logspace(-3, 0, 4), "gamma": np.logspace(-2, 2, 5)}, cv=3)
            regressor.fit(X_rainy_days, y_rainy_days)

        elif methodName == 'RF':
            regressor = GridSearchCV(RandomForestRegressor(), param_grid={"max_depth": [20, 60]}, cv=3)
            regressor.fit(X_rainy_days, y_rainy_days)

        elif methodName == 'XGB':
            regressor = XGBRegressor(n_estimators=100, max_depth=6, early_stopping_rounds=20, learning_rate=0.1)
            X_train, X_valid, y_train, y_valid = train_test_split(X_rainy_days, y_rainy_days, test_size=0.2)
            regressor.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=0)
            history_reg = regressor.evals_result()

        elif methodName == 'ANN':
            # sklearn implementation
            # regressor = MLPRegressor(hidden_layer_sizes=(200,), max_iter=100000)
            # regressor = GridSearchCV(MLPRegressor( max_iter=100000),
            #                          param_grid={'hidden_layer_sizes': [5,10,20,50,100,200]}, cv=3)
            # regressor.fit(X_rainy_days, y_rainy_days)

            # keras tensorflow implementation
            regressor = tf.keras.models.Sequential([
                keras.layers.Dense(8, activation='relu', input_shape=X_rainy_days.shape[1:]),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(1), ])
            regressor.compile(optimizer='adam', loss='mse', metrics=['mse'])
            history_reg = regressor.fit(X_rainy_days, y_rainy_days, epochs=100, validation_split=.2, verbose=0,
                    callbacks=tf_nn_callbacks)

        elif methodName in ['CNN', ]:
            # Prepare shape for convolution
            X_rainy_days = np.swapaxes(np.swapaxes(X_rainy_days, 1, 2), 2, 3)
            nfilters = 32
            regressor = tf.keras.models.Sequential()
            if X.shape[1] < 5 or X.shape[2] < 5:
                regressor.add(layers.Conv2D(filters=nfilters, kernel_size=(1, 1), activation='relu', input_shape=X.shape[1:]))
            else:
                regressor.add(layers.Conv2D(filters=nfilters, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
            # if X.shape[1] >= 10 and X.shape[2] >= 10:
            #     regressor.add(layers.MaxPooling2D(pool_size=2))
            #     regressor.add(layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
            regressor.add(layers.Flatten())
            regressor.add(layers.Dense(nfilters, activation='relu'))
            regressor.add(layers.Dense(8, activation='relu'))
            regressor.add(layers.Dense(1))
            regressor.compile(optimizer='adam', loss='mse', metrics=['mse'])
            history_reg = regressor.fit(X_rainy_days, y_rainy_days, epochs=100, validation_split=.2, verbose=0, callbacks=tf_nn_callbacks)

    # Plot hyperparameters
    if plot_hyperparameters_epochs_nEstimators_featureImportances == True:
        if targetVar == 'pr':
            plot.hyperparameters_epochs_nEstimators_featureImportances(classifier, targetVar, methodName, ipoint, 'classifier', history_clf)
        plot.hyperparameters_epochs_nEstimators_featureImportances(regressor, targetVar, methodName, ipoint, 'regressor', history_reg)


    return regressor, classifier

########################################################################################################################

if __name__=="__main__":

    nproc = MPI.COMM_WORLD.Get_size()         # Size of communicator
    iproc = MPI.COMM_WORLD.Get_rank()         # Ranks in communicator
    inode = MPI.Get_processor_name()          # Node where this MPI process runs
    targetVar = sys.argv[1]
    methodName = sys.argv[2]
    family = sys.argv[3]
    mode = sys.argv[4]
    fields = sys.argv[5]

    train_chunk(targetVar, methodName, family, mode, fields, iproc, nproc)
    MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step