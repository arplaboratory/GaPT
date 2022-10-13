"""
Residual lenarning SISO

"""

import logging
import json
import sys
import yaml
import math
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from configs import OUTPUT_DIR
from base.data.data_loader import GPDataLoader
from configs.config_parsers import ArgParser, LoggerParser, MatplotlibParser
from model import datasets
from model import regressors
from tools.file_utils import Folder
from sklearn.metrics import mean_squared_error

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

# SESSION NAME
session_name = session_id

# Fix the Seed for numpy and pytorch
np.random.seed(seeds_config["numpy"])
torch.manual_seed(seeds_config["torch"])

# OUTPUT FOLDER AND LOGGING
run_id = ''.join([session_name, '_', datetime.now().strftime(r'%m%d_%H%M%S')])
session_folder = Folder(OUTPUT_DIR, run_id)
folder_csv_out = Folder(session_folder.path, 'data_out')
folder_json_out = Folder(session_folder.path, 'configuration')
folder_figures_out = Folder(session_folder.path, 'figures')
folder_yamlcpp_out = Folder(session_folder.path, 'yaml')
folder_model_out = Folder(session_folder.path, 'model')

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

# Get the parameters of the training dataset
dataset_args_tr = dataset_config_train["train"]["args"]
dataset_model_tr = dataset_config_train["train"]["class"]
dataset_class_tr = getattr(datasets, dataset_model_tr)

# Instantiate the training dataset
training_dataset = dataset_class_tr(**dataset_args_tr)
training_unscaled = dataset_class_tr(**dataset_args_tr)

# TEST

# Get the parameters of the training dataset
dataset_args_ts = dataset_config_train["test"]["args"]
dataset_model_ts = dataset_config_train["test"]["class"]
dataset_class_ts = getattr(datasets, dataset_model_ts)

# Instantiate the training dataset
testing_dataset = dataset_class_ts(**dataset_args_ts)
testing_unscaled = dataset_class_ts(**dataset_args_ts)

#############################################
#           NORMALIZE DATASETS              #
#############################################

# Get the mean and std
mean_y_tr = torch.mean(training_dataset.Y)
std_y_tr = torch.std(training_dataset.Y)

mean_y_ts = torch.mean(testing_dataset.Y)
std_y_ts = torch.std(testing_dataset.Y)

# normalization on the values
# L2 norm on the rpms (last four columns of the dataset)
# Z score on the first column and the

# TRAINING

mean_vx_tr = torch.mean(training_dataset.X[:, 0], dim=0)
std_vx_tr = torch.std(training_dataset.X[:, 0], dim=0)
training_dataset.X[:, 0] = (training_dataset.X[:, 0] - mean_vx_tr) / std_vx_tr
training_dataset.Y = (training_dataset.Y - mean_y_tr) / std_y_tr

# TESTING
mean_vx_ts = torch.mean(testing_dataset.X[:, 0], dim=0)
std_vx_ts = torch.std(testing_dataset.X[:, 0], dim=0)
testing_dataset.X[:, 0] = (testing_dataset.X[:, 0] - mean_vx_ts) / std_vx_ts
testing_dataset.Y = (testing_dataset.Y - mean_y_ts) / std_y_ts


#############################################
#    REGRESSOR TRAINING                     #
#############################################

# Define the dataloader
k_points = gpytorch_config["k_training_points"]
train_dataloader = GPDataLoader(training_dataset, k_training_points=k_points)

# Create the model from the config file
regressor_name = regressors_config['used'][0]  # We use just one regressor
model_class = getattr(regressors, regressor_name)
model = model_class(id_model="Residual_Regressor", input_dim=1)

# Train the model using GpyTorch
model.train_hyperparams(train_dataloader=train_dataloader, gp_training_conf=gpytorch_config)


#############################################
#           PREDICTION GPYTORCH             #
#############################################

# TRAINING
gpt_pred_mean_tr, gpt_pred_lower_tr, gpt_pred_upper_tr = model.predict_gpyt(training_dataset.X)

gpt_pred_mean_tr = mean_y_tr + gpt_pred_mean_tr * std_y_tr
gpt_pred_lower_tr = gpt_pred_lower_tr * std_y_tr
gpt_pred_upper_tr = gpt_pred_upper_tr * std_y_tr

# TEST
gpt_pred_mean_ts, gpt_pred_lower_ts, gpt_pred_upper_ts = model.predict_gpyt(testing_dataset.X)

gpt_pred_mean_ts = mean_y_tr + gpt_pred_mean_ts * std_y_tr
gpt_pred_lower_ts = gpt_pred_lower_ts * std_y_tr
gpt_pred_upper_ts = gpt_pred_upper_ts * std_y_tr

#############################################
#               PREDICTIONS KF              #
#############################################

#  in GpyTorch the data are stored and sorted. we do
#  the same thing here -> We sort the data (speed
#  and residual) according the values of the speed.

# TRAINING
# get the training dataset
x_tr_kal = training_dataset.X.squeeze().numpy()
y_tr_kal = training_dataset.Y.squeeze().numpy()
t_tr_kal = training_dataset.timestamp.numpy()

x_tr_kal = x_tr_kal[1200:2400]
y_tr_kal = y_tr_kal[1200:2400]
t_tr_kal = t_tr_kal[1200:2400]

# a) Get the indexes to sorting elements in the dataset (if used the regressor works as the same as Gpytorch)
sort_index_tr = np.argsort(x_tr_kal)


# b) Get the rollback indexes
rollback_indexes_tr = np.empty_like(sort_index_tr)
rollback_indexes_tr[sort_index_tr] = np.arange(sort_index_tr.size, dtype=int)

x_tr_kal_sorted = x_tr_kal[sort_index_tr]
y_tr_kal_sorted = y_tr_kal[sort_index_tr]
t_tr_kal_sorted = t_tr_kal[sort_index_tr]

kf_pred_mean_tr, kf_pred_lower_tr, kf_pred_upper_tr = model.predict_kf_mean_cov(
    x_training=x_tr_kal_sorted, y_training=y_tr_kal_sorted,
    z_mean=mean_y_tr.numpy(),
    z_std=std_y_tr.numpy(),
    x_test=training_dataset.X.numpy(),
    Qgain=0.01)


# TEST
x_ts_kal = testing_dataset.X.squeeze().numpy()
y_ts_kal = testing_dataset.Y.squeeze().numpy()
t_ts_kal = testing_dataset.timestamp.squeeze().numpy()

kf_pred_mean_ts, kf_pred_lower_ts, kf_pred_upper_ts = model.predict_kf_mean_cov(
    x_training=x_tr_kal_sorted, y_training=y_tr_kal_sorted,
    z_mean=mean_y_ts.numpy(),
    z_std=std_y_ts.numpy(),
    x_test=testing_dataset.X.numpy(), Qgain=0.1)


#########################################
#              FIGURES                  #
#########################################

min_y_tr = torch.min(training_unscaled.Y)
max_y_tr = torch.max(training_unscaled.Y)
y_lim_tr = [min_y_tr + (0.1 * min_y_tr), max_y_tr + (0.1 * max_y_tr)]

min_y_ts = torch.min(testing_unscaled.Y)
max_y_ts = torch.max(testing_unscaled.Y)
y_lim_ts = [min_y_ts + (0.1 * min_y_ts), max_y_ts + (0.1 * max_y_ts)]


##############
# DATASET    #
##############

# Define the dataloader
train_dataloader = GPDataLoader(training_unscaled, k_training_points=k_points)

# get the x and Y training just to display figures
x_train_gp, y_train_gp, ts_train_gp = next(iter(train_dataloader))

fig, ax = plt.subplots(1, 1)
ax.set_title(r"Sampled Pts: $v_y$ VS. $\tilde{a}_y$")
ax.plot(training_unscaled.X, training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(x_train_gp, y_train_gp, color=matplot_colors[1], marker=matplot_mark_style[5],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="Training")
ax.set_xlabel(r"$v_y$ $\mathrm{[m/s]}$")
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "siso_training_data_1" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

fig, ax = plt.subplots(1, 1)
ax.set_title(r"Sampled Pts: $time$ VS. $\tilde{a}_y$")
ax.plot(training_unscaled.timestamp, training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(ts_train_gp, y_train_gp, color=matplot_colors[1], marker=matplot_mark_style[5],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="Training")
ax.set_xlabel(r"$elapsed$ $time$ $\mathrm{[s]}$")
ax.set_ylabel(r"$v_y$ $\mathrm{[m/s]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "siso_training_data_2" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

##############
# TRAINING   #
##############

fig, ax = plt.subplots(1, 1)
ax.set_title(r"Training dataset: $v_y$ VS. $\tilde{a}_y$")
ax.plot(training_unscaled.X, training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(training_unscaled.X, kf_pred_mean_tr, color=matplot_colors[3], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4],
        label=r"$\tilde{a}_{y_{KF}}$")
ax.plot(training_unscaled.X, gpt_pred_mean_tr, color=matplot_colors[1], marker=matplot_mark_style[8],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[0],
        label=r"$\tilde{a}_{y_{GPT}}$")
ax.set_xlabel(r"$v_y$ $\mathrm{[m/s]}$")
ax.set_ylabel(r"$\tilde{a}_y^t$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "siso_KF_pred_train_1" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

fig, ax = plt.subplots(1, 1)
ax.set_title(r"Training dataset: $time$ VS. $\tilde{a}_y$")
ax.plot(training_unscaled.timestamp, training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(training_unscaled.timestamp, kf_pred_mean_tr, color=matplot_colors[3], marker=matplot_mark_style[8],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[0],
        label=r"$\tilde{a}_{y_{KF}}$")
ax.plot(training_unscaled.timestamp, gpt_pred_mean_tr, color=matplot_colors[1], marker=matplot_mark_style[8],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[0],
        label=r"$\tilde{a}_{y_{GPT}}$")
ax.fill_between(training_unscaled.timestamp, kf_pred_upper_tr, kf_pred_lower_tr, alpha=0.5, color=matplot_colors[3])
ax.set_xlabel(r"$elapsed$ $time$ $\mathrm{[s]}$")
ax.set_ylabel(r"$\tilde{a}_y^t$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "siso_KF_pred_train_2" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

##############
# TEST       #
##############
fig, ax = plt.subplots(1, 1)
ax.set_title(r"Testing dataset: $V_y$ VS. $\tilde{a}_y$")
ax.plot(testing_unscaled.X, testing_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(testing_unscaled.X, kf_pred_mean_ts, color=matplot_colors[3], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4],
        label=r"$\tilde{a}_{y_{KF}}$")
ax.plot(testing_unscaled.X, gpt_pred_mean_ts, color=matplot_colors[1], marker=matplot_mark_style[8],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[0],
        label=r"$\tilde{a}_{y_{GPT}}$")
ax.set_xlabel(r"$v_y$ $\mathrm{[m/s]}$")
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_ts)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "siso_KF_pred_test_1" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

fig, ax = plt.subplots(1, 1)
ax.set_title(r"Testing dataset: $time$ VS. $\tilde{a}_y$")
ax.plot(testing_unscaled.timestamp, testing_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(testing_unscaled.timestamp, kf_pred_mean_ts, color=matplot_colors[3], marker=matplot_mark_style[8],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[0],
        label=r"$\tilde{a}_{y_{KF}}$")
ax.plot(testing_unscaled.timestamp, gpt_pred_mean_ts, color=matplot_colors[1], marker=matplot_mark_style[8],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[0],
        label=r"$\tilde{a}_{y_{GPT}}$")
ax.fill_between(testing_unscaled.timestamp, kf_pred_upper_ts, kf_pred_lower_ts, alpha=0.5, color=matplot_colors[3])
ax.set_xlabel(r"$elapsed$ $time$ $\mathrm{[s]}$")
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_ts)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "siso_KF_pred_test_2" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

##############################
# OUTPUTS                    #
##############################
# - Save Gptmodel
model.dump_model(folder_model_out.path)

# - Create the dataframe
columns_tr = [training_unscaled.t_column, training_unscaled.x_column[0], training_unscaled.y_column,
              "gpt_pred_mean_tr", "gpt_pred_lower_tr", "gpt_pred_upper_tr",
              "kf_pred_mean_tr", "kf_pred_lower_tr", "kf_pred_upper_tr"]
val_tr = [training_unscaled.timestamp.numpy(), training_unscaled.X.squeeze().numpy(),
          training_unscaled.Y.squeeze().numpy(),
          gpt_pred_mean_tr.numpy(), gpt_pred_lower_tr.numpy(), gpt_pred_upper_tr.numpy(),
          kf_pred_mean_tr, kf_pred_lower_tr, kf_pred_upper_tr]

columns_tr_used = ["t_tr_sort_scal_KF", "x_tr_sort_scal_KF", "y_tr_sort_scal_KF", 'desorting_indexes']
val_tr_used = [t_tr_kal_sorted, x_tr_kal_sorted, y_tr_kal_sorted, rollback_indexes_tr]

columns_ts = [testing_unscaled.t_column, testing_unscaled.x_column[0], testing_unscaled.y_column,
              "gpt_pred_mean_ts", "gpt_pred_lower_ts", "gpt_pred_upper_ts",
              "kf_pred_mean_ts", "kf_pred_lower_ts", "kf_pred_upper_ts"]
val_ts = [testing_unscaled.timestamp.numpy(), testing_unscaled.X.squeeze().numpy(),
          testing_unscaled.Y.squeeze().numpy(),
          gpt_pred_mean_ts.numpy(), gpt_pred_lower_ts.numpy(), gpt_pred_upper_ts.numpy(),
          kf_pred_mean_ts, kf_pred_lower_ts, kf_pred_upper_ts]

data_frame_tr = pd.DataFrame(columns=columns_tr)
data_tr = np.vstack(val_tr).T
data_frame_tr[columns_tr] = data_tr
data_frame_tr.reset_index(drop=True)
np.testing.assert_array_equal(data_tr[:, 0], val_tr[0][:], err_msg='', verbose=True)

data_frame_tr_used = pd.DataFrame(columns=columns_tr_used)
data_tr_used = np.vstack(val_tr_used).T
data_frame_tr_used[columns_tr_used] = data_tr_used
data_frame_tr_used.reset_index(drop=True)

data_frame_ts = pd.DataFrame(columns=columns_ts)
data_ts = np.vstack(val_ts).T
data_frame_ts[columns_ts] = data_ts
data_frame_ts.reset_index(drop=True)
np.testing.assert_array_equal(data_ts[:, 0], val_ts[0][:], err_msg='', verbose=True)

# - Save the dataframe as .csv
data_frame_tr.to_csv(Path(folder_csv_out.path, 'out_train.csv'), sep=',', index=False)
data_frame_tr_used.to_csv(Path(folder_csv_out.path, 'out_train_used.csv'), sep=',', index=False)
data_frame_ts.to_csv(Path(folder_csv_out.path, 'out_test.csv'), sep=',', index=False)

# - Save the config file as .json
with open(Path(folder_json_out.path, 'residual_arpl_siso.json'), 'w') as outfile:
    json.dump(json_config, outfile)

# - Create the yamlfile
discrete_model = model._pssgp_cov.get_sde()
P_inf = discrete_model[0].numpy().flatten().tolist()
F = discrete_model[1].numpy().flatten().tolist()
L = discrete_model[2].numpy().flatten().tolist()
H = discrete_model[3].numpy().flatten().tolist()
Q = (discrete_model[4].numpy()).flatten().tolist()

xtrain_yaml = x_tr_kal.flatten().tolist()
ytrain_yaml = y_tr_kal.flatten().tolist()
R = float(model._kf.R)
Order = int(model._order)
meanY_yaml = np.mean(training_unscaled.Y.numpy())
std_yaml = np.std(training_unscaled.Y.numpy())
dict_file = {"axis": "y_drag",
             "y_drag": {
                 "R": R,
                 "Order": int(model._order),
                 "x_fact": 1,
                 "P_inf": P_inf,
                 "F": F,
                 "L": L,
                 "H": H,
                 "Q": Q,
                 "xtrain": x_tr_kal_sorted,
                 "ytrain": y_tr_kal_sorted,
                 "meanY": float(meanY_yaml),
                 "std": float(std_yaml),
             }
             }

with open(Path(folder_yamlcpp_out.path, 'model_params.yaml'), 'w', ) as outfile:
    documents = yaml.dump(dict_file, outfile, )
