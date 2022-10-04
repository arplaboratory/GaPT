"""
The script will evaluate the computational time for one sample inference for both the GPyTorch and the
KF models.
miso
"""
import copy
import gc
import logging
import numpy
import time
import numpy as np
import pandas as pd
import torch.random
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from base.data.data_loader import GPDataLoader
from configs import OUTPUT_DIR
from configs.config_parsers import ArgParser, LoggerParser, MatplotlibParser
from model import datasets
from model import regressors
from tools.file_utils import Folder


##############################
# LOAD CONFIG                #
##############################
parser = ArgParser()
config = parser.args
json_config = parser.json

dataset_config_train = config.DATASET
regressors_config = config.REGRESSORS
seeds_config = config.SEEDS
gpytorch_config = config.GPT_MODEL
session_id = config.SESSION_NAME
log_config = config.LOG
kpts_config = config.KPOINTS
measurmements_config = config.N_MEASUREMENTS

# SESSION NAME
session_name = session_id

# Fix the Seed for numpy and pytorch
np.random.seed(seeds_config["numpy"])
torch.manual_seed(seeds_config["torch"])

# OUTPUT FOLDERS
run_id = ''.join([session_name, '_RT_MISO', datetime.now().strftime(r'%m%d_%H%M%S')])
session_folder = Folder(OUTPUT_DIR, run_id)
folder_csv_out = Folder(session_folder.path, 'inf_time_benchmark')
folder_json_out = Folder(session_folder.path, 'configuration')

# LOGGING
LoggerParser.setup_logging(save_dir=session_folder.path)
logging.info("Session: {} started".format(session_name))

# MATLAB
matplt_conf = MatplotlibParser.get_matplot_conf()
plt.rcParams["figure.figsize"] = (matplt_conf["figsize"][0], matplt_conf["figsize"][1])
matplotlib.rc("font", **matplt_conf["font"])
matplotlib.rc("axes", **matplt_conf["axes"])
matplotlib.rc("figure", **matplt_conf["figure"])
matplotlib.rc("xtick.major", **matplt_conf["xtick"]["major"])
matplotlib.rc("ytick.major", **matplt_conf["ytick"]["major"])
matplotlib.rcParams["pdf.fonttype"] = matplt_conf["pdf_fonttype"]
matplotlib.rcParams["ps.fonttype"] = matplt_conf["ps_fonttype"]
matplot_colors = matplt_conf["colors"]
matplot_line_tick = matplt_conf["line_thickness"]
matplot_mark_size = matplt_conf["marker_size"]
matplot_line_style = matplt_conf["extra_linestyles"]
matplot_mark_style = matplt_conf["markers"]

#############################################
#           LOAD THE DATASET                #
#############################################

# TRAINING

# TRAINING
# Get the parameters of the training dataset
dataset_args_tr = dataset_config_train["train"]["args"]
dataset_model_tr = dataset_config_train["train"]["class"]
dataset_class_tr = getattr(datasets, dataset_model_tr)

# Instantiate the training dataset
training_dataset = dataset_class_tr(**dataset_args_tr)

############################################
#           NORMALIZE DATASET               #
#############################################

# Get the mean and std
mean_y_tr = torch.mean(training_dataset.Y)
std_y_tr = torch.std(training_dataset.Y)

# Z score on the first column and the
for i in range(training_dataset.X[:, :].shape[1]):
    std_x_0_tr, mean_x_0_tr = torch.std_mean(training_dataset.X[:, i])
    if i > 0:
        std_x_0_tr = std_x_0_tr * 10
    training_dataset.X[:, i] = (training_dataset.X[:, i] - mean_x_0_tr) / std_x_0_tr
training_dataset.Y = (training_dataset.Y - mean_y_tr) / std_y_tr

#############################################
#               PREDICTIONS KF              #
#############################################
# TRAIN

# get the training dataset
x_tr_kal = training_dataset.X.numpy()
y_tr_kal = training_dataset.Y.numpy()

# Sort the indexes
index_tr = np.argsort(x_tr_kal[:, 0])

# Get the rollback indexes
rollback_indexes_tr = np.empty_like(index_tr)
rollback_indexes_tr[index_tr] = np.arange(index_tr.size)

# Sort the training dataset used for the prediction matching
x_tr_kal_sort = x_tr_kal[index_tr]
y_tr_kal_sort = y_tr_kal[index_tr]

##############################
# K-PTS LIST                 #
##############################

# DEFINE POINTS
pt_list = kpts_config["pt_list"]
k_training_pts_list = np.array(pt_list)
##############################
# BENCHMARK                  #
##############################

# Create a list of models
pred_list = []

# Get the list of model used for the test
regressor_list = regressors_config['used']
for regressor_name in regressor_list:
    model_class = getattr(regressors, regressor_name)
    model = model_class(id_model="",  input_dim=len(training_dataset.x_column))
    model_order = None
    model_name = model.__name__.split("_")[0]
    pred_list.append([model_name, model, np.zeros((len(k_training_pts_list), 2)), model_order])

for ix_pts in tqdm(range(0, len(k_training_pts_list)), desc="Iterating K pts.", colour="green", leave=True):
    k_pts = k_training_pts_list[ix_pts]

    # 1) Create a Dataloader
    dataloader = GPDataLoader(training_dataset, k_training_points=int(k_pts))

    # 2) Train the models and perform the predictions for both the GPyTorch and the KF implementation
    for model_ix in tqdm(range(0, len(pred_list)), desc="Processing Models", colour="blue", leave=False):

        model = copy.deepcopy(pred_list[model_ix][1])
        # a) Train the regressor with the dataloader
        model.train_hyperparams(dataloader, gpytorch_config)

        # b) Get the dimension of m matrix of the trained model and the got model to perform the predictions
        pred_list[model_ix][3] = model._pssgp_cov.get_sde()[1].shape[
            0]  # TODO: ACCESS TO PRIVATE MEMBER, MAYBE CREATE A GETTER

        gpt_kernel, gpt_likelihood, gpt_model = model.get_gpt_models()

        # c) Extract the dataset points used for the benchmark
        x_test_pt = training_dataset.X[0:k_pts+1]
        y_test_pt = training_dataset.Y[0:k_pts+1]

        # b) predict with the GPytorch just one sample model and get the pred time, if the num. of the samples
        # are more, pytorch will parallelize the computation and the comparison with the sequential Kalman filter
        # wouldn't be fair.

        # c) Perform multiple measurments, remove the min,max values e store the mean values
        n_measurements = measurmements_config["n_measurments"]
        if n_measurements < 3:
            msg = "The number of measurements required to estimate the inference time must be >10. Received:{}" \
                .format(n_measurements)
            logging.error(msg)
            raise ValueError(msg)

        pred_time_gpt = numpy.zeros(n_measurements)
        pred_time_kf = numpy.zeros(n_measurements)

        for n_meas in tqdm(range(n_measurements),
                           desc=" GpyTorch: Performing multiple measurements", colour="purple", leave=False):
            # ca) perform inference with GPyTorch
            # single point test (first)
            x_val = x_test_pt[0].unsqueeze(1).T
            t_now = time.time()
            with torch.no_grad():
                _ = gpt_likelihood(gpt_model(x_val))
                pred_time_gpt[n_meas] = time.time() - t_now

        del model._gpt_kernel, model._gpt_likelihood, model._gpt_model

        for n_meas in tqdm(range(n_measurements),
                           desc=" KF: Performing multiple measurements", colour="yellow", leave=False):
            # cb) perform inference with Kernels
            t_now = time.time()
            _, _, _ = model.predict_kf_miso(x_training=x_tr_kal_sort[0:k_pts+1],
                                            y_training=y_tr_kal_sort[0:k_pts+1],
                                            z_mean=mean_y_tr.numpy(),
                                            z_std=std_y_tr.numpy(),
                                            x_test=x_test_pt.numpy()[0],
                                            reversed=False)

            pred_time_kf[n_meas] = time.time() - t_now

        pred_time_gpt = np.delete(pred_time_gpt, [np.argmax(pred_time_gpt), np.argmin(pred_time_gpt)])
        pred_time_kf = np.delete(pred_time_kf, [np.argmax(pred_time_kf), np.argmin(pred_time_kf)])

        pred_list[model_ix][2][ix_pts][0] = np.mean(pred_time_gpt)
        pred_list[model_ix][2][ix_pts][1] = np.mean(pred_time_kf)

        del model, gpt_kernel, gpt_likelihood, gpt_model
        gc.collect()

    del dataloader
    gc.collect()

##############################
# OUTPUTS                    #
##############################
# - Create the dataframe
data_frame = pd.DataFrame(columns=['K_pts'])
data_frame['K_pts'] = k_training_pts_list
for model in pred_list:
    data_frame[model[0] + "_GPY"] = model[2][:, 0]
    data_frame[model[0] + "_KF"] = model[2][:, 1]

# - Save the dataframe as .csv
data_frame.to_csv(Path(folder_csv_out.path, 'results.csv'), sep=',')

# - Save the config file as .json
with open(Path(folder_json_out.path, 'inference_time_arpl.json'), 'w') as outfile:
    json.dump(json_config, outfile)

