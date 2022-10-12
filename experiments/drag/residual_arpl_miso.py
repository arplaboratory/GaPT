"""
Residual lenarning MISO

"""
import logging
import json
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


##############################
# LOAD CONFIG                #
##############################
parser = ArgParser()
config = parser.args
json_config = parser.json

dataset_config = config.DATASET
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
dataset_args_tr = dataset_config["train"]["args"]
dataset_model_tr = dataset_config["train"]["class"]
dataset_class_tr = getattr(datasets, dataset_model_tr)

# Instantiate the training dataset
training_dataset = dataset_class_tr(**dataset_args_tr)
training_unscaled = dataset_class_tr(**dataset_args_tr)

# TEST
# Get the parameters of the training dataset
dataset_args_ts = dataset_config["test"]["args"]
dataset_model_ts = dataset_config["test"]["class"]
dataset_class_ts = getattr(datasets, dataset_model_ts)

# Instantiate the testing dataset
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


# TRAINING
for i in range(training_dataset.X[:, :].shape[1]):
    std_x_0_tr, mean_x_0_tr = torch.std_mean(training_dataset.X[:, i])
    if i > 0:
        std_x_0_tr = std_x_0_tr*10
    training_dataset.X[:, i] = (training_dataset.X[:, i] - mean_x_0_tr) / std_x_0_tr
training_dataset.Y = (training_dataset.Y - mean_y_tr) / std_y_tr


# TESTING
for i in range(testing_dataset.X[:, :].shape[1]):
    std_x_0_ts, mean_x_0_ts = torch.std_mean(testing_dataset.X[:, i])
    testing_dataset.X[:, i] = (testing_dataset.X[:, i] - mean_x_0_ts) / std_x_0_ts
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
model = model_class(id_model="Residual_Regressor", input_dim=len(training_dataset.x_column))

# Train the model using GpyTorch
model.train_hyperparams(train_dataloader=train_dataloader, gp_training_conf=gpytorch_config)

############################################
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
# TRAIN

# get the training dataset
x_tr_kal = training_dataset.X.numpy()
y_tr_kal = training_dataset.Y.numpy()

# Select the training subset used for the kalman filter
x_tr_kal = x_tr_kal[1200:4800]
y_tr_kal = y_tr_kal[1200:4800]


# Sort the indexes
index_tr = np.argsort(x_tr_kal[:, 0])



# Get the rollback indexes
rollback_indexes_tr = np.empty_like(index_tr)
rollback_indexes_tr[index_tr] = np.arange(index_tr.size)

# Sort the training dataset used for the prediction matching
x_tr_kal_sort = x_tr_kal[index_tr]
y_tr_kal_sort = y_tr_kal[index_tr]

# Predict
kf_pred_mean_tr, kf_pred_lower_tr, kf_pred_upper_tr = model.predict_kf_miso(x_training=x_tr_kal_sort,
                                                                            y_training=y_tr_kal_sort,
                                                                            z_mean=mean_y_tr.numpy(),
                                                                            z_std=std_y_tr.numpy(),
                                                                            x_test=training_dataset.X.numpy(),
                                                                            reversed=False)


# TEST
kf_pred_mean_ts, kf_pred_lower_ts, kf_pred_upper_ts = model.predict_kf_miso(x_training=x_tr_kal_sort,
                                                                            y_training=y_tr_kal_sort,
                                                                            z_mean=mean_y_ts.numpy(),
                                                                            z_std=std_y_ts.numpy(),
                                                                            x_test=testing_dataset.Y.numpy(),
                                                                            reversed=True)



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
train_un_dataloader = GPDataLoader(training_unscaled, k_training_points=k_points)

# get the x and Y training just to display figures
x_train_gp, y_train_gp, ts_train_gp = next(iter(train_un_dataloader))

# t - res_vy
fig, ax = plt.subplots(1, 1)
ax.set_title(r"Sampled Pts: $time$ VS. $\tilde{a}_y$")
ax.plot(training_unscaled.timestamp, training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(ts_train_gp, y_train_gp, color=matplot_colors[1], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size-1, linestyle=matplot_line_style[4], label="Training")
ax.set_xlabel(r"$elapsed$ $time$ $\mathrm{[s]}$")
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "miso_training_t_vy" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

# vy - res_vy
fig, ax = plt.subplots(1, 1)
ax.set_title(r"Sampled Pts: $v_y$ VS. $\tilde{a}_y$")
ax.plot(training_unscaled.X[:, 0], training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(x_train_gp[:, 0], y_train_gp, color=matplot_colors[1], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size-1, linestyle=matplot_line_style[4], label="Training")
ax.set_xlabel(r"$v_y$ $\mathrm{[m/s]}$")
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "miso_training_vy" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")


# u1 - res_vy
fig, ax = plt.subplots(1, 1)
ax.set_title(r"Sampled Pts: Motor 1 VS. $\tilde{a}_y$")
ax.plot(training_unscaled.X[:, 1], training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(x_train_gp[:, 1], y_train_gp, color=matplot_colors[1], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size-1, linestyle=matplot_line_style[4], label="Training")
ax.set_xlabel(r"Motor 1 speed $\mathrm{[rpm]}$")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "miso_training_u1" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

# u2 - res_vy
fig, ax = plt.subplots(1, 1)
ax.set_title(r"Sampled Pts: Motor 2 VS. $\tilde{a}_y$")
ax.plot(training_unscaled.X[:, 2], training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(x_train_gp[:, 2], y_train_gp, color=matplot_colors[1], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size-1, linestyle=matplot_line_style[4], label="Training")
ax.set_xlabel(r"Motor 2 speed $\mathrm{[rpm]}$")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "miso_training_u2" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

# u3 - res_vy
fig, ax = plt.subplots(1, 1)
ax.set_title(r"Sampled Pts: Motor 3 VS. $\tilde{a}_y$")
ax.plot(training_unscaled.X[:, 3], training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(x_train_gp[:, 3], y_train_gp, color=matplot_colors[1], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size-1, linestyle=matplot_line_style[4], label="Training")
ax.set_xlabel(r"Motor 3 speed $\mathrm{[rpm]}$")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "miso_training_u3" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")


# u4 - res_vy
fig, ax = plt.subplots(1, 1)
ax.set_title(r"Sampled Pts: Motor 4 VS. $\tilde{a}_y$")
ax.plot(training_unscaled.X[:, 4], training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(x_train_gp[:, 4], y_train_gp, color=matplot_colors[1], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size-1, linestyle=matplot_line_style[4], label="Training")
ax.set_xlabel(r"Motor 4 speed $\mathrm{[rpm]}$")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "miso_training_u4" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

##############
# TRAINING   #
##############

fig, ax = plt.subplots(1, 1)
ax.set_title(r"Training dataset: $v_y$ VS. $\tilde{a}_y$")
ax.plot(training_unscaled.X[:, 0], training_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(training_unscaled.X[:, 0], kf_pred_mean_tr, color=matplot_colors[3], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4],
        label=r"$\tilde{a}_{y_{KF}}$")
ax.plot(training_unscaled.X[:, 0], gpt_pred_mean_tr, color=matplot_colors[1], marker=matplot_mark_style[8],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[0],
        label=r"$\tilde{a}_{y_{GPT}}$")
ax.set_xlabel(r"$v_y$ $\mathrm{[m/s]}$")
ax.set_ylabel(r"$\tilde{a}_y^t$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_tr)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "miso_KF_pred_train_1" + ".pdf")
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
path = Path(folder_figures_out.path, "miso_KF_pred_train_2" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")

##############
# TEST       #
##############
fig, ax = plt.subplots(1, 1)
ax.set_title(r"Testing dataset: $time$ VS. $\tilde{a}_y$")
ax.plot(testing_unscaled.X[:, 0], testing_unscaled.Y, color=matplot_colors[0], marker=matplot_mark_style[2],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4], label="GT")
ax.plot(testing_unscaled.X[:, 0], kf_pred_mean_ts, color=matplot_colors[3], marker=matplot_mark_style[7],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[4],
        label=r"$\tilde{a}_{y_{KF}}$")
ax.plot(testing_unscaled.X[:, 0], gpt_pred_mean_ts, color=matplot_colors[1], marker=matplot_mark_style[8],
        linewidth=matplot_line_tick, markersize=matplot_mark_size, linestyle=matplot_line_style[0],
        label=r"$\tilde{a}_{y_{GPT}}$")
ax.set_xlabel(r"$v_y^t$ $\mathrm{[m/s]}$")
ax.set_ylabel(r"$\tilde{a}_y$ $\mathrm{[m/s^2]}$")
ax.grid(alpha=0.3)
ax.set_ylim(y_lim_ts)
ax.legend(loc="best", markerscale=5, fontsize=15)
fig.tight_layout()
path = Path(folder_figures_out.path, "miso_KF_pred_test_1" + ".pdf")
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
path = Path(folder_figures_out.path, "miso_KF_pred_test_2" + ".pdf")
fig.savefig(path, format="pdf", bbox_inches="tight")


##############################
# OUTPUTS                    #
##############################
# - Save Gptmodel
model.dump_model(folder_model_out.path)

# - Create the dataframes
# TRAIN
columns_tr = ["gpt_pred_mean_tr", "gpt_pred_lower_tr", "gpt_pred_upper_tr",
              "kf_pred_mean_tr", "kf_pred_lower_tr", "kf_pred_upper_tr"]
val_tr = [gpt_pred_mean_tr.numpy(), gpt_pred_lower_tr.numpy(), gpt_pred_upper_tr.numpy(),
          kf_pred_mean_tr, kf_pred_lower_tr, kf_pred_upper_tr]

columns_tr.append(training_unscaled.t_column)
val_tr.append(training_unscaled.timestamp.numpy())

for i in range(len(training_unscaled.x_column)):
    columns_tr.append(training_unscaled.x_column[i])
    val_tr.append(training_unscaled.X[:, i].squeeze().numpy())

columns_tr.append(training_unscaled.y_column)
val_tr.append(training_unscaled.Y.numpy())

# test dataset
columns_ts = ["gpt_pred_mean_ts", "gpt_pred_lower_ts", "gpt_pred_upper_ts",
              "kf_pred_mean_ts", "kf_pred_lower_ts", "kf_pred_upper_ts"]
val_ts = [gpt_pred_mean_ts.numpy(), gpt_pred_lower_ts.numpy(), gpt_pred_upper_ts.numpy(),
          kf_pred_mean_ts, kf_pred_lower_ts, kf_pred_upper_ts]

columns_ts.append(testing_unscaled.t_column)
val_ts.append(testing_unscaled.timestamp.numpy())

for i in range(len(testing_unscaled.x_column)):
    columns_ts.append(testing_unscaled.x_column[i])
    val_ts.append(testing_unscaled.X[:, i].squeeze().numpy())

columns_ts.append(testing_unscaled.y_column)
val_ts.append(testing_unscaled.Y.numpy())

data_frame_tr = pd.DataFrame(columns=columns_tr)
data_tr = np.asarray(val_tr).T
data_frame_tr[columns_tr] = data_tr
data_frame_tr.reset_index(drop=True)

data_frame_ts = pd.DataFrame(columns=columns_ts)
data_ts = np.asarray(val_ts).T
data_frame_ts[columns_ts] = data_ts
data_frame_ts.reset_index(drop=True)

# - Save the dataframe as .csv
data_frame_tr.to_csv(Path(folder_csv_out.path, 'out_train.csv'), sep=',', index=False)
data_frame_ts.to_csv(Path(folder_csv_out.path, 'out_test.csv'), sep=',', index=False)

# - Save the config file as .json
with open(Path(folder_json_out.path, 'residual_arpl_miso_rpms.json'), 'w') as outfile:
    json.dump(json_config, outfile)
