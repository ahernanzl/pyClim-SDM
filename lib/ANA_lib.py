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
def train_PCA():
    """
    Train PCA of sinoptic analogy fields preserving n_components needed to explain exp_var_ratio_th (advanced_settings)
    """

    # Define pathOut
    pathOut = pathAux + 'PCA/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Get synoptic fields and weights
    saf_train = np.load(pathAux + 'STANDARDIZATION/SAF/'+target_vars0[0]+'_training.npy')
    saf_train = saf_train.astype('float32')

    # Prepare data for PCA
    ndays, nsafs, nlats, nlons = saf_train.shape[0], saf_train.shape[1], saf_train.shape[2], saf_train.shape[3]
    saf_train = saf_train.reshape(ndays, -1)
    W = W_saf[np.newaxis, :]
    W = np.repeat(W, ndays, axis=0)
    saf_train *= W

    # Perform PCA
    pca = PCA(exp_var_ratio_th)
    pca.fit(saf_train)

    # Save trained pca object
    outfile = open(pathOut + 'pca', 'wb')
    pickle.dump(pca, outfile)
    outfile.close()


########################################################################################################################
def set_number_of_weather_types():
	"""
	Performs cluster for different values of k_clusters and saves elbow_curve to decide parameter k_cluster and define
	it at settings.
	It is performed with MiniBatchKmeans, which is much faster than kmeans but a bit less accurate.
	This is only used to set the number of clusters, accuracy is not crucial.
	Nevertheless, it is prepared for Kmeans by setting the parameter "clustering_method".

	Some parameters of both MiniBatchKmeans and Kmeans might need to be tuned.
	"""

	print('weather_types: set number of clusters')
	# Define clustering method
	# clustering_method = 'Kmeans'
	clustering_method = 'MiniBatchKmeans'

	# Get synoptic fields and weights
	saf_train = np.load(pathAux+'STANDARDIZATION/SAF/p_training.npy')
	saf_train = saf_train.astype('float32')

	# Prepare data for PCA
	ndays, nsafs, nlats, nlons = saf_train.shape[0], saf_train.shape[1], saf_train.shape[2], saf_train.shape[3]
	saf_train = saf_train.reshape(ndays, -1)
	W = W_saf[np.newaxis, :]
	W = np.repeat(W, ndays, axis=0)
	saf_train *= W

	# Load pca object
	infile = open(pathAux + 'PCA/pca', 'rb')
	pca = pickle.load(infile)
	infile.close()

	# Transform sat_train to PCA
	saf_train = pca.transform(saf_train)

	# Chose the number of clusters: k.
	Nc = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 500, 50)) + list(range(500, 1000, 100)) + list(range(1000, 2000, 200))
	score = []
	for i in Nc:
		start_i = datetime.datetime.now()

		convergence = False
		max_iter = 500
		while convergence == False:
			if clustering_method == 'Kmeans':
				kmeans = KMeans(n_clusters=i, max_iter=max_iter).fit(saf_train)
			elif clustering_method == 'MiniBatchKmeans':
				kmeans = MiniBatchKMeans(n_clusters=i, init_size=i, max_iter=max_iter).fit(saf_train)
			n_iter = kmeans.n_iter_ + 1
			if n_iter == max_iter:
				max_iter += 100
			else:
				convergence = True

		score.append(kmeans.score(saf_train))

		print('------------')
		print('max_iter', max_iter)
		print('n_iter', n_iter)
		print('k =', i, '\nElapsed time: ' + str(datetime.datetime.now() - start_i))

	# Saves elbow_curve figure
	pathOut=pathAux+'WEATHER_TYPES/'
	if not os.path.exists(pathOut):
		os.makedirs(pathOut)
	plt.plot(Nc, score)
	plt.xlabel('Number of Clusters: k')
	plt.ylabel('Score')
	plt.title('Elbow Curve')
	plt.savefig(pathOut + 'Elbow_Curve.png')
	plt.close()
	exit('See ' + pathOut + '/Elbow_Curve.png and set k_clusters at settings')



########################################################################################################################
def get_weather_types_centroids():
	"""
	Cluster by k-mean method.
	Outputs: centroids and labels and distances from days to their centroids

	It is performed with Kmeans, which is more accurate than MiniBatchKmeans.
	Nevertheless it is prepared for MiniBatchKmeans.

	Some parameters of both MiniBatchKmeans and Kmeans might need to be tuned.
	"""

	print('weather_types: get centroids')

	# Define clustering method
	clustering_method = 'Kmeans'
	# clustering_method = 'MiniBatchKmeans'

	# Get synoptic fields and weights
	saf_train = np.load(pathAux+'STANDARDIZATION/SAF/'+target_vars0[0]+'_training.npy')
	saf_train = saf_train.astype('float32')

	# Prepare data for PCA
	ndays, nsafs, nlats, nlons = saf_train.shape[0], saf_train.shape[1], saf_train.shape[2], saf_train.shape[3]
	saf_train = saf_train.reshape(ndays, -1)
	W = W_saf[np.newaxis, :]
	W = np.repeat(W, ndays, axis=0)
	saf_train *= W

	# Load pca object
	infile = open(pathAux + 'PCA/pca', 'rb')
	pca = pickle.load(infile)
	infile.close()

	# Transform sat_train to PCA
	saf_train = pca.transform(saf_train)

	# Read number of clusters
	k = k_clusters
	print('k_clusters =', k)

	# Perform the kmeans clustering until convergence is achieved
	convergence = False
	max_iter = 300
	while convergence == False:
		print('------------')
		print('max_iter', max_iter)
		if clustering_method == 'Kmeans':
			kmeans = KMeans(n_clusters=k, max_iter=max_iter).fit(saf_train)
		elif clustering_method == 'MiniBatchKmeans':
			kmeans = MiniBatchKMeans(n_clusters=k, init_size=k, max_iter=max_iter).fit(saf_train)
		n_iter = kmeans.n_iter_+1
		print('n_iter', n_iter)
		if n_iter == max_iter:
			max_iter+=50
		else:
			convergence = True
		print('convergence:', convergence)

	# Calculate centroids, labels and dist from days to their centroids
	centroids = kmeans.cluster_centers_
	labels = kmeans.labels_
	dist = np.sqrt(np.average((saf_train - centroids[labels]) ** 2, axis=1))

	# Plot number of elements in each cluster
	pathOut = pathAux+'WEATHER_TYPES/'
	if not os.path.exists(pathOut):
		os.makedirs(pathOut)
	plt.hist(labels, bins=k)
	plt.xlabel('Cluster id')
	plt.ylabel('Elements of the cluster')
	plt.title('Numer of clusters: ' + str(k))
	# plt.show()
	# exit()
	plt.savefig(pathOut+'Elements_per_cluster_k' + str(k) + '.png')
	plt.close()

	# Write centroids and labels to files
	np.save(pathOut + '/centroids', centroids)
	np.save(pathOut + '/labels', labels)
	np.save(pathOut + '/dist', dist)


########################################################################################################################
def get_weather_type_id(scene, centroids):
    """
    This function determines the weather type of the particular date
    :param scene: (ndays, n_syn_anal_fields, nlats, nlons)
    :param centroids: (k_clusters, n_syn_anal_fields, nlats, nlons)
    :param W: (n_syn_anal_fields, nlats, nlons)
    :return: the id of the weather type and the distance to the centroid
    """

    # Format to scene and centroids
    scene = np.repeat(scene, k_clusters, 0)
    scene = scene.reshape(scene.shape[0], -1)
    centroids = centroids.reshape(centroids.shape[0], -1)

    # Calculate distances and get minimum
    dist = np.sqrt(np.average((scene-centroids)**2, axis=1))
    k_index = np.argsort(dist)[0]

    return {'k_index': k_index, 'dist': dist.min()}


########################################################################################################################
def get_synoptic_distances(calib, scene):
    '''
    Returns array of synoptic distance for each calibration day.
    :param calib: (ndays, nPC)
    :param scene: (1, nPC)
    :return: dist: (ndays)
    '''

    # Format to scene and calib
    scene = np.repeat(scene, calib.shape[0], 0)
    scene = scene.reshape(scene.shape[0], -1)
    calib = calib.reshape(calib.shape[0], -1)

    # Calculate distances
    dist = np.sqrt(np.average((scene-calib)**2, axis=1))

    return dist


########################################################################################################################
def get_local_distances(pred_calib, pred_scene, ipred):
    """
    Returns array of local distance for each calibration day and point.
    :param pred_calib: (n_analogs_preselection, npreds)
    :param pred_scene:  (, npreds)
    :param ipred:
    :return: dist: (n_analogs_preselection,)
    """

    # Format to pred_calib and pred_scene
    pred_calib = pred_calib[:,ipred]
    pred_scene = pred_scene[:,ipred]
    pred_scene = np.repeat(pred_scene, pred_calib.shape[0], 0)

    # Calculate local distance
    dist = np.sqrt(np.mean((pred_scene - pred_calib)**2, axis=1))
    # print len(np.where(dist<0.75)[0])

    return dist


########################################################################################################################
def coefficients(var, methodName, mode, iproc=0, nproc=1):
    """
    """

    mode = 'PP'
    var0 = var[0]

    # Define pathOut
    pathOut = '../tmp/cluster_' + '_'.join(((var, methodName))) + '/'
    if not os.path.exists(pathOut) and iproc==0:
        os.makedirs(pathOut)
    MPI.COMM_WORLD.Barrier()  # Waits for all subprocesses to complete last step

    # Read metadata of hres grid and the neighbour associated to each point
    i_4nn = np.load(pathAux+'ASSOCIATION/'+var0.upper()+'_'+interp_mode+'/i_4nn.npy')
    j_4nn = np.load(pathAux+'ASSOCIATION/'+var0.upper()+'_'+interp_mode+'/j_4nn.npy')
    w_4nn = np.load(pathAux+'ASSOCIATION/'+var0.upper()+'_'+interp_mode+'/w_4nn.npy')

    # Read synoptic analogy fields and centroids
    pred = np.load(pathAux+'STANDARDIZATION/PRED/t_training.npy')
    saf_train = np.load(pathAux+'STANDARDIZATION/SAF/t_training.npy')
    centroids = np.load(pathAux+'WEATHER_TYPES/centroids.npy')

    # Prepare data for PCA
    ndays, nsafs, nlats, nlons = saf_train.shape[0], saf_train.shape[1], saf_train.shape[2], saf_train.shape[3]
    saf_train = saf_train.reshape(ndays, -1)
    W = W_saf[np.newaxis, :]
    W = np.repeat(W, ndays, axis=0)
    saf_train *= W

    # Load pca object
    infile = open(pathAux + 'PCA/pca', 'rb')
    pca = pickle.load(infile)
    infile.close()

    # Transform sat_train to PCA
    saf_train = pca.transform(saf_train)

    # Read high resolution data and transform to int to save memory and to be homogeneous with downscale scene
    if iproc == 0:
        obs = read.hres_data(var, period='training')['data']
        obs = (100 * obs).astype(predictands_codification[var]['type'])
    else:
        obs = None
    MPI.COMM_WORLD.Barrier()  # Waits for all subprocesses to complete last step

    if nproc > 1:
        obs = MPI.COMM_WORLD.bcast(obs, root=0)
    special_value = 100*predictands_codification[var]['special_value']

    # Create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(k_clusters) / n_chunks))
    ik = [i for i in range(k_clusters)]

    k_chunk = []
    for ichunk in range(n_chunks):
        k_chunk.append(ik[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    len_chunk = []
    for ichunk in range(n_chunks):
        len_chunk.append(len(k_chunk[ichunk]))

    # Create empty array to accumulate correlation coefficients
    coef = np.zeros((len_chunk[iproc], hres_npoints[var[0]], n_preds_t))
    intercept = np.zeros((len_chunk[iproc], hres_npoints[var[0]], 1))

    # Go through k clusters
    for ik in range(len_chunk[iproc]):
        k_global = k_chunk[iproc][ik]

        print(var, methodName, 'coefficients. k=', k_global,' (', round(100*ik/len_chunk[iproc]), '%)')

        # Searches synoptic analogs to the centroid
        i_centroid = centroids[k_global][np.newaxis, :]
        dist = ANA_lib.get_synoptic_distances(saf_train, i_centroid)
        iana = np.argsort(dist)[:n_analogs_preselection]

        # Selects analogs only with certain amount of precipitation
        obs_array=obs[iana, :]
        pred_array=pred[iana]

        # Calculate partial correlations for each point and predictor
        for ipoint in range(hres_npoints):
            Y = obs_array[:, ipoint]
            valid = np.where(Y < special_value)[0]

            # If not enough data for calibration, fill with np.nan
            if valid.size < 150:
                print('Not enough valid predictands')
                coef[ik, ipoint] = np.nan
                intercept[ik, ipoint] = np.nan
            else:
                X = pred_array[valid, :, :, :]
                Y = Y[valid]

                # Create predictors array of analog days to the cluster centroid, by selecting the nearest neighbour or by
                # interpolating the 4 neighbouts, depending on the setting parameter "n_neighbours"
                X = grids.interpolate_predictors(X, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_mode)
                regressor = RidgeCV()
                regressor.fit(X, Y)
                coef[ik, ipoint] = regressor.coef_
                intercept[ik, ipoint] = regressor.intercept_

    # Save coefficients
    np.save(pathOut+'coef_ichunk_' + str(iproc), coef)
    np.save(pathOut+'intercept_ichunk_' + str(iproc), intercept)


########################################################################################################################
def coefficients_collect_chunks(var, methodName, mode, nproc=1):

    # Define pathOut
    pathOut=pathAux+'COEFFICIENTS/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    n_chunks = nproc

    print('--------------------------------------')
    print(var, methodName, 'cluster collect chunks', n_chunks)

    # Create empty array and accumulate
    coef=np.zeros((0, hres_npoints, n_preds_t))
    intercept=np.zeros((0, hres_npoints, 1))
    for ichunk in range(n_chunks):
        path = '../tmp/cluster_' + '_'.join(((var, methodName))) + '/'
        filename = path + 'coef_ichunk_' + str(ichunk) + '.npy'
        coef = np.append(coef, np.load(filename), axis=0)
        filename = path + 'intercept_ichunk_' + str(ichunk) + '.npy'
        intercept = np.append(intercept, np.load(filename), axis=0)
    shutil.rmtree(path)

    np.save(pathOut+var+'_'+methodName+'_coefficients', coef)
    np.save(pathOut+var+'_'+methodName+'_intercept', intercept)

########################################################################################################################
def correlations(var, methodName, mode, iproc=0, nproc=1, th_metric='median'):
    """
    Searches significant predictors for each grid point and weather type.

    For each weather type selects all the synoptic analogs (those given by dist_th, which correspond to th_metric of the
    train set).
    Then, for each grid point calculates partial correlations with all possible predictors.
    Correlation will be computed only using precipitation data greater than "pcp_th_for_corr".
    Correlation will be computed only when having more than "min_days_corr".
    Nevertheless no regression will be performed. These predictors will be used only to search analogy in them.

    """

    mode = 'PP'
    var0 = var[0]

    # Define pathOut
    pathOut = '../tmp/cluster_' + '_'.join(((var, methodName))) + '/'
    if not os.path.exists(pathOut) and iproc==0:
        os.makedirs(pathOut)

    MPI.COMM_WORLD.Barrier()  # Waits for all subprocesses to complete last step

    pcp_th_for_corr = 0.1 # mm

    # Read metadata of hres grid and the neighbour associated to each point
    i_4nn = np.load(pathAux+'ASSOCIATION/'+var0.upper()+'_'+interp_mode+'/i_4nn.npy')
    j_4nn = np.load(pathAux+'ASSOCIATION/'+var0.upper()+'_'+interp_mode+'/j_4nn.npy')
    w_4nn = np.load(pathAux+'ASSOCIATION/'+var0.upper()+'_'+interp_mode+'/w_4nn.npy')

    # Read pred, saf and centroids
    pred = np.load(pathAux+'STANDARDIZATION/PRED/p_training.npy')
    saf_train = np.load(pathAux+'STANDARDIZATION/SAF/p_training.npy')
    centroids = np.load(pathAux+'WEATHER_TYPES/centroids.npy')

    # Prepare data for PCA
    ndays, nsafs, nlats, nlons = saf_train.shape[0], saf_train.shape[1], saf_train.shape[2], saf_train.shape[3]
    saf_train = saf_train.reshape(ndays, -1)
    W = W_saf[np.newaxis, :]
    W = np.repeat(W, ndays, axis=0)
    saf_train *= W

    # Load pca object
    infile = open(pathAux + 'PCA/pca', 'rb')
    pca = pickle.load(infile)
    infile.close()

    # Transform sat_train to PCA
    saf_train = pca.transform(saf_train)

    # Read high resolution data and transform to int to save memory
    if iproc == 0:
        obs=read.hres_data(var, period='training')['data']
        obs = (100 * obs).astype(predictands_codification[var]['type'])
    else:
        obs = None
    MPI.COMM_WORLD.Barrier()  # Waits for all subprocesses to complete last step

    if nproc>1:
        obs = MPI.COMM_WORLD.bcast(obs, root=0)

    special_value = 100*predictands_codification[var]['special_value']


    # Create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(k_clusters) / n_chunks))
    ik = [i for i in range(k_clusters)]

    k_chunk = []
    for ichunk in range(n_chunks):
        k_chunk.append(ik[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    len_chunk = []
    for ichunk in range(n_chunks):
        len_chunk.append(len(k_chunk[ichunk]))

    # Create empty array to accumulate correlation coefficients
    R = np.zeros((len_chunk[iproc], hres_npoints, n_preds_p))

    # Get dist_th
    if th_metric == 'median':
        dist_th = np.median(np.load(pathAux + 'WEATHER_TYPES/dist.npy'))
    elif th_metric == 'max':
        dist_th = np.max(np.load(pathAux + 'WEATHER_TYPES/dist.npy'))
    elif th_metric == 'p90':
        dist_th = np.percentile(np.load(pathAux + 'WEATHER_TYPES/dist.npy'), 90)

    # Go through k clusters
    for ik in range(len_chunk[iproc]):
        k_global = k_chunk[iproc][ik]

        # Searches synoptica analogs to the centroid
        i_centroid = centroids[k_global][np.newaxis, :]
        dist = ANA_lib.get_synoptic_distances(saf_train, i_centroid)
        iana = np.where(dist < dist_th)[0]
        print('k =', k_global, ',', iana.size, 'days  (', round(100*ik/len_chunk[iproc]), '%)')

        # Selects analogs only with certain amount of precipitation
        obs_array=obs[iana, :]
        pred_array=pred[iana]

        # Calculate partial correlations for each point and predictor
        for ipoint in range(hres_npoints):

            # Select only rainy data
            Y = obs_array[:, ipoint]
            valid = np.where((Y > 100. * pcp_th_for_corr) * (Y < special_value))[0]
            X = pred_array[valid, :, :, :]
            Y = Y[valid]

            # If not enogh data to establish correlation Nan
            if Y.size < min_days_corr:
                R[ik, ipoint, :] = np.nan
            else:
                # Create predictors array of analog days to the cluster centroid, by selecting the nearest neighbour or by
                # interpolating the 4 neighbouts, depending on the setting parameter "n_neighbours"
                X = grids.interpolate_predictors(X, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_mode)
                for ipred in range(n_preds_p):
                    R[ik, ipoint, ipred] = spearmanr(X[:, ipred], Y)[0]

    # Save results
    np.save(pathOut+'ichunk_' + str(iproc), R)



########################################################################################################################
def correlations_collect_chunks(var, methodName, mode, nproc=1):

    # Define pathOut
    pathOut=pathAux+'COEFFICIENTS/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    n_chunks = nproc

    print('--------------------------------------')
    print(var, methodName, 'cluster collect chunks', n_chunks)

    # Create empty array and accumulate
    R = np.zeros((0, hres_npoints, n_preds_p))
    for ichunk in range(n_chunks):
        path = '../tmp/cluster_' + '_'.join(((var, methodName))) + '/'
        filename = path + 'ichunk_' + str(ichunk) + '.npy'
        R = np.append(R, np.load(filename), axis=0)
    shutil.rmtree(path)

    np.save(pathOut+'pcp_ANA_correlations', R)


########################################################################################################################
if __name__ == "__main__":
    nproc = MPI.COMM_WORLD.Get_size()  # Size of communicator
    iproc = MPI.COMM_WORLD.Get_rank()  # Ranks in communicator
    inode = MPI.Get_processor_name()  # Node where this MPI process runs

    var = sys.argv[1]
    methodName = sys.argv[2]
    mode = sys.argv[3]
    func = sys.argv[4]

    if func == 'correlations':
        correlations(var, methodName, mode, iproc, nproc)
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
        if iproc==0:
            correlations_collect_chunks(var, methodName, mode, nproc)
    elif func == 'coefficients':
        coefficients(var, methodName, mode, iproc, nproc)
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
        if iproc==0:
            coefficients_collect_chunks(var, methodName, mode, nproc)
