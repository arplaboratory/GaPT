# GaPT - Gaussian Process Toolkit for Online Regression

## License
Please be aware that this code was originally implemented for research purposes and may be subject to changes and any fitness for a particular purpose is disclaimed. To inquire about commercial licenses, please contact Prof. Giuseppe Loianno (loiannog@nyu.edu).
```
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 ```
## Citation
If you publish a paper with this work, please cite our paper: 
```
@article{CrocettiMaoICRA2023,
  url = {https://arxiv.org/abs/2303.08181},
  Year = {2023},
  Booktitle = {IEEE International Conference on Robotics and Automation (ICRA)}
  author = {Crocetti, Francesco and Mao, Jeffrey and Saviolo, Alessandro and Costante, Gabriele and Loianno, Giuseppe},
  title = {GaPT: Gaussian Process Toolkit for Online Regression with Application to Learning Quadrotor Dynamics}}
 
## Overview:

GaPT  is a novel toolkit that converts GPs to their state space form and performs regression in
linear time. 
- GaPT is designed to be highly compatible with several optimizers popular in robotics
GaPT is able to capture the system behavior in multiple flight regimes and
operating conditions, including those producing highly nonlinear effects such as aerodynamic forces and rotor interactions.
- GaPT has a faster inference time (processing sample rate) compared to a classical GP inference approach on both single and multi-input settings especially
when considering large number of data points processed sample by sample (online).


### Disclaimer: the repository is work in progress
That means we are working on uploading and upgrading all the functionalities. For now, it is intended to show the code
which is the core of the GaPT Framework.

#### Recent Changes
- 07/26/2023: due to known security vulnerabilities, the requirements.txt file has been modified to install newer versions of 
  - Scipy ( 1.8.1 -> 1.10.1) 
  - pytorch (1.10.1 -> 1.13.1)
- 10/04/2022: First commit, the requirements and the setup.py file will be updated soon: 
the installation will be automated, now the file requirements.txt is just a freeze of the original virtual environment.
- Upcoming updates: Installation procedure for desktop x86 devices and Nvidia ARM64 architecture devices. 
## Installation
The following installation procedure uses Anaconda3, that allow to create different and isolated python environments.
If you don't have it, please see the installation guide: https://www.anaconda.com/products/distribution .

### Create the environment
Go into the Installation folder and run the following command to create an empty environment
config file.
```shell
conda create -n GAPT python=3.8
```
Before installing the required python packages, **the environment must be activated**.  
**NOTE:** this action is required whenever you want to work with the desired environment.

### Activate the environment:  

```shell
conda activate GAPT
```

### Install the requirements
Be sure to have the latest version of Pip and install the requirements.
```shell
pip install --upgrade pip
conda install -c conda-forge empy
pip install -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

**NOTE:** this action must be performed from the installation folder and with the venv active.

## Testing the installation:
The '*experiments*' folder contains two subdirectories with the example code to use the GaPt and to perform 
some benchmarks. Note that each file needs the relative .json config file that can be passed as argument.

**NOTE:** this action must be performed from the Root folder of the repository and with the venv active.

To run the experiment, run the following command from the root of the project:
```shell
python -m experiments.<folder>.<experiments_name> -c <path_to_config_file>.json. 
```
For example to run the test for the SISO regression:
```shell
python -m experiments.drag.residual_arpl_siso -c experiments/drag/residual_arpl_siso.json
```
**NOTE:** If you run the code using and IDE (i.e. Pycharm) the paths in the .json config file may be changed 
using the absolute path and the path of the config file with the -c flag have to be included in the 'Parameter'
field of the script configuration.
SISO Regression scripts runs both ours toolboxes output and the standard Pytorch output for a comparison 

The output of the scripts are in the /out foulder. They include the following 

Foulder Name | Contents
------------- | -------------
Configuration  | The parameter you used to run the python file 
data_out | the output of the data in the form of .csv
figures | Figures demonstrating the output of the Toolbox. 
Model | A .pth file including the weights of the GaPT Model
Yaml | A .yaml model file to export to ROS. Only SISO so far. 


Acronyms      | Meaning
------------- | -------------
GT            | Ground Truth
Kf            | Toolbox Output
GPT           | GPytorch baseline output
