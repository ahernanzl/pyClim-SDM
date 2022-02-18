import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import BC_lib
import down_scene_ANA
import down_scene_BC
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_GCMs
import evaluate_methods
import grids
import launch_jobs
import plot
import postpro_lib
import postprocess
import derived_predictors
import preprocess
import process
import read
import standardization
import TF_lib
import val_lib
import WG_lib
import write


########################################################################################################################
def train_chunk(var, methodName, family, mode, fields, iproc=0, nproc=1):
    '''
    Calibrates regression for all points,divided in chunks if run at HPC.
    '''

    # Define pathOut
    pathOut='../tmp/TRAINING_'  + var + '_' + methodName + '/'

    # Declares variables for father process, who creates pathOut
    if iproc == 0:
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)
        pred_calib = None
        var_calib = None
        if 'pred' in fields:
            pred_calib = np.load(pathAux+'STANDARDIZATION/PRED/'+var[0]+'_training.npy')
            pred_calib = pred_calib.astype('float32')
        if 'var' in fields:
            var_calib = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_training.npy')
        obs = read.hres_data(var, period='training')['data']
        obs = (100 * obs).astype(predictands_codification[var]['type'])
        i_4nn = np.load(pathAux+'ASSOCIATION/'+interp_dict[mode]+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+interp_dict[mode]+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+interp_dict[mode]+'/w_4nn.npy')

    # Declares variables for the other processes
    else:
        pred_calib = None
        var_calib = None
        obs = None
        i_4nn = None
        j_4nn = None
        w_4nn = None

    # Share data with all subprocesses
    if nproc > 1:
        pred_calib = MPI.COMM_WORLD.bcast(pred_calib, root=0)
        var_calib = MPI.COMM_WORLD.bcast(var_calib, root=0)
        obs = MPI.COMM_WORLD.bcast(obs, root=0)
        i_4nn = MPI.COMM_WORLD.bcast(i_4nn, root=0)
        j_4nn = MPI.COMM_WORLD.bcast(j_4nn, root=0)
        w_4nn = MPI.COMM_WORLD.bcast(w_4nn, root=0)


    # create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(hres_npoints) / n_chunks))
    points_chunk = []
    for ichunk in range(n_chunks):
        points_chunk.append(list(range(hres_npoints))[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    ichunk = iproc
    npoints_ichunk = len(points_chunk[ichunk])

    regressors = npoints_ichunk * [None]
    classifiers = npoints_ichunk * [None]
    if get_reg_and_clf_scores == True:
        regressors_scores = np.zeros(npoints_ichunk)
        classifiers_scores = np.zeros(npoints_ichunk)

    # If we are tuning hyperparameters only certaing points will be calculated
    if plot_hyperparameters == True:
        points_chunk[ichunk]=[x for x in points_chunk[ichunk] if x%500==0]

    # loop through all points of the chunk
    special_value = 100 * predictands_codification[var]['special_value']
    for ipoint in points_chunk[ichunk]:
        ipoint_local_index = points_chunk[ichunk].index(ipoint)
        if ipoint_local_index % 1 == 0:
            print('--------------------')
            print('ichunk:	', ichunk, '/', n_chunks)
            print('training', var, methodName, round(100*ipoint_local_index/npoints_ichunk, 2), '%')

        # Get preds from neighbour/s and trains model for echa point
        Y = obs[:, ipoint]
        valid = np.where(Y < special_value)[0]
        if valid.size < 30:
            exit('Not enough valid predictands to train')
        Y = Y[valid]
        X = pred_calib[valid, :, :, :]
        X = grids.interpolate_predictors(X, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_dict[mode])

        # Train TF (clf and reg)
        regressors[ipoint_local_index], classifiers[ipoint_local_index] = train_ipoint(var, methodName, X, Y, ipoint)
        if get_reg_and_clf_scores == True:
            regressors_scores[ipoint_local_index] = regressors[ipoint_local_index].score(X, Y)
            if var == 'pcp':
                classifiers_scores[ipoint_local_index] = classifiers[ipoint_local_index].score(X, Y)



    # Save chunks
    if plot_hyperparameters == False:
        outfile = open(pathOut + 'regressors_'+str(ichunk), 'wb')
        pickle.dump(regressors, outfile)
        outfile.close()
        outfile = open(pathOut + 'classifiers_'+str(ichunk), 'wb')
        pickle.dump(classifiers, outfile)
        outfile.close()
        if get_reg_and_clf_scores == True:
            np.save(pathOut + 'regressors_scores_'+str(ichunk), regressors_scores)
            if var == 'pcp':
                np.save(pathOut + 'classifiers_scores_'+str(ichunk), classifiers_scores)


########################################################################################################################
def collect_chunks(var, methodName, family, n_chunks=1):
    """
    This function collects the results of downscale_chunk() and saves them into a final single file.
    """

    print('--------------------------------------')
    print('collect chunks', n_chunks)

    # Define paths
    pathIn = '../tmp/TRAINING_' + var + '_' + methodName + '/'
    pathOut = pathAux + 'TRAINED_MODELS/' + var.upper() + '/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Create empty regressors/classifiers to be filled
    regressors = np.zeros((0, ))
    classifiers = np.zeros((0, ))
    if get_reg_and_clf_scores == True:
        regressors_scores = np.zeros((0, ))
        if var == 'pcp':
            classifiers_scores = np.zeros((0, ))

    # Read trained models chunks and accumulate them
    for ichunk in range(n_chunks):
        infile = open(pathIn+'regressors_'+str(ichunk), 'rb')
        regressors_ichunk = pickle.load(infile)
        infile.close()
        infile = open(pathIn+'classifiers_'+str(ichunk), 'rb')
        classifiers_ichunk = pickle.load(infile)
        infile.close()
        regressors = np.append(regressors, regressors_ichunk, axis=0)
        classifiers = np.append(classifiers, classifiers_ichunk, axis=0)
        if get_reg_and_clf_scores == True:
            regressors_scores_ichunk = np.load(pathIn + 'regressors_scores_' + str(ichunk) + '.npy')
            regressors_scores = np.append(regressors_scores, regressors_scores_ichunk, axis=0)
            if var == 'pcp':
                classifiers_scores_ichunk = np.load(pathIn + 'classifiers_scores_' + str(ichunk) + '.npy')
                classifiers_scores = np.append(classifiers_scores, classifiers_scores_ichunk, axis=0)
    os.system('rm -r ' + pathIn)

    # Save to file
    outfile = open(pathOut + methodName + '_reg', 'wb')
    pickle.dump(regressors, outfile)
    outfile.close()
    if get_reg_and_clf_scores == True:
        np.save(pathOut + methodName + '_R2_score', regressors_scores)

    if var == 'pcp':
        outfile = open(pathOut + methodName + '_clf', 'wb')
        pickle.dump(classifiers, outfile)
        outfile.close()
        if get_reg_and_clf_scores == True:
            np.save(pathOut + methodName + '_acc_score', classifiers_scores)


########################################################################################################################
def train_ipoint(var, methodName, X, Y, ipoint):
    '''
    Train model (classifiers and regressors)
    '''

    classifier = None
    regressor = None

    # For precipitation trains classier+regression, but for temperature only regression
    if var[0] == 't':

        # Regressor t
        if methodName == 'MLR':
            regressor = RidgeCV(cv=3)
        elif methodName == 'ANN':
            # regressor = GridSearchCV(MLPRegressor(activation='logistic', hidden_layer_sizes=(70,), solver='sgd',
            #                                       learning_rate='adaptive', max_iter=10000),
            #                                         param_grid={'alpha': [0.0001, 0.001, 0.01, 0.1]}, cv=3)
            # regressor = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100000)
            # regressor = GridSearchCV(MLPRegressor(max_iter=100000), param_grid={'alpha': [0.00001,0.0001, 0.001, 0.01, 0.1,1,10]}, cv=3)
            # regressor = MLPRegressor(hidden_layer_sizes=(200,), max_iter=100000)
            regressor = GridSearchCV(MLPRegressor(max_iter=100000),
                                     param_grid={'hidden_layer_sizes': [5,10,20,50,100,200]}, cv=3)
        elif methodName == 'SVM':
            regressor = GridSearchCV(svm.SVR(kernel='rbf'),
                                param_grid={"C": np.logspace(3, 5, 3), "gamma": np.logspace(-3, 0, 3)}, cv=3)
        elif methodName == 'LS-SVM':
            regressor = GridSearchCV(KernelRidge(kernel='rbf'),
                                 param_grid={"alpha": np.logspace(-3, 0, 4), "gamma": np.logspace(-2, 2, 5)}, cv=3)


        # For LS-SVM and SVR uses a random sample of 5000 data
        if methodName in ('SVM', 'LS-SVM'):
        # if regressor_name in ('LS-SVM', 'SVR', 'MLPR'):
            nDays = X.shape[0]
            nRand = min(nDays, 5000)
            rng = np.random.default_rng()
            iDays = rng.choice(nDays, size=nRand, replace=False)
            X, Y = X[iDays], Y[iDays]
        regressor.fit(X, Y)
        # if methodName in ('SVM', 'LS-SVM'):
        #     print(regressor.best_params_)

    else:
        israiny = (Y > (100 * wetDry_th))
        nDays = Y.size
        # If all data are dry, clasiffiers won't work. A random data is forced into wet.
        minWetDays = 3
        allDry = False
        if len(np.where(israiny==False)[0]) == nDays:
            allDry = True
        if allDry == True:
            rand = random.sample(range(nDays), minWetDays)
            israiny[rand] = True
            Y[rand] = 100*0.1
        X_rainy_days = X[israiny==True]
        if allDry == True:
            for i in range(minWetDays):
                X_rainy_days[i] = 999 - i
        Y_rainy_days = Y[israiny==True]

        # Classifier pcp
        if methodName[:3] == 'GLM':
            classifier = LogisticRegressionCV(cv=3, max_iter=1000)
            # classifier = RidgeClassifierCV(cv=3)
        elif methodName == 'ANN':
            # classifier = GridSearchCV(MLPClassifier(activation='logistic', hidden_layer_sizes=(10,), solver='sgd',
            #                                       learning_rate='adaptive', max_iter=10000),
            #                                         param_grid={'alpha': [0.1, 1, 10]}, cv=3)
            # classifier = MLPClassifier(hidden_layer_sizes=(200,), max_iter=100000)
            classifier = GridSearchCV(MLPClassifier(max_iter=100000),
                                     param_grid={'hidden_layer_sizes': [5,10,20,50,100,200]}, cv=3)
        elif methodName == 'SVM':
            classifier = GridSearchCV(svm.SVC(kernel='rbf'),
                                    param_grid={"C": np.logspace(0, 1, 2), "gamma": np.logspace(-2, -1, 2)}, cv=3)
        elif methodName == 'LS-SVM':
            classifier = RidgeClassifierCV(cv=3)
        elif methodName == 'RF':
            # classifier = RandomForestClassifier()
            classifier = GridSearchCV(RandomForestClassifier(),
                                    param_grid={"max_depth": [20, 50, 100]}, cv=3)

        # Transform classifier to CalibratedClassifierCV to get probabilities from classes.
        # if classifier_mode == 'probabilistic':
        classifier = CalibratedClassifierCV(classifier, cv=5)

        if methodName == 'SVM':
            nDays = X.shape[0]
            nRand = min(nDays, 7500)
            rng = np.random.default_rng()
            iDays = rng.choice(nDays, size=nRand, replace=False)
            classifier.fit(X[iDays], 1*israiny[iDays])
        else:
            # Fit
            classifier.fit(X, 1*israiny)

        # Regressor t
        if methodName == 'GLM-LIN':
            regressor = RidgeCV(cv=3)
        elif methodName == 'GLM-EXP':
            Y_rainy_days = np.log(Y_rainy_days)
            regressor = RidgeCV(cv=3)
        elif methodName == 'GLM-CUB':
            Y_rainy_days = np.cbrt(Y_rainy_days)
            regressor = RidgeCV(cv=3)
        elif methodName == 'ANN':
            # regressor = GridSearchCV(MLPRegressor(activation='logistic', hidden_layer_sizes=(10,), solver='sgd',
            #                                       learning_rate='adaptive', max_iter=10000),
            #                                         param_grid={'alpha': [0.1, 1, 10]}, cv=3)
            # regressor = MLPRegressor(hidden_layer_sizes=(200,), max_iter=100000)
            regressor = GridSearchCV(MLPRegressor( max_iter=100000),
                                     param_grid={'hidden_layer_sizes': [5,10,20,50,100,200]}, cv=3)
        elif methodName == 'SVM':
            regressor = GridSearchCV(svm.SVR(kernel='rbf'),
                                param_grid={"C": np.logspace(3, 5, 3), "gamma": np.logspace(-2, 0, 3)}, cv=3)
        elif methodName == 'LS-SVM':
            regressor = GridSearchCV(KernelRidge(kernel='rbf'),
                                 param_grid={"alpha": np.logspace(-3, 0, 4), "gamma": np.logspace(-2, 2, 5)}, cv=3)
        elif methodName == 'RF':
            # regressor = RandomForestRegressor()
            regressor = GridSearchCV(RandomForestRegressor(),
                                    param_grid={"max_depth": [20, 50, 100]}, cv=3)


        # Fit
        regressor.fit(X_rainy_days, Y_rainy_days)
        # if methodName == 'LS-SVM':
        #     print(regressor.best_params_)

        # Plot hyperparameters
        if plot_hyperparameters == True:
            plot.hyperparameters(regressor, methodName, ipoint)

    return regressor, classifier

########################################################################################################################

if __name__=="__main__":

    nproc = MPI.COMM_WORLD.Get_size()         # Size of communicator
    iproc = MPI.COMM_WORLD.Get_rank()         # Ranks in communicator
    inode = MPI.Get_processor_name()          # Node where this MPI process runs
    var = sys.argv[1]
    methodName = sys.argv[2]
    family = sys.argv[3]
    mode = sys.argv[4]
    fields = sys.argv[5]

    train_chunk(var, methodName, family, mode, fields, iproc, nproc)
    MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
    if iproc==0:
        collect_chunks(var, methodName, family, nproc)