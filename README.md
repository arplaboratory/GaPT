# GaPT - Gaussian Process Toolkit for Online Regression

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
- 10/04/2022: First commit, the requirements and the setup.py file will be updated soon: 
the installation will be automated, now the file requirements.txt is just a freeze of the original virtual environment.
- Upcoming updates: Installation procedure for desktop x86 devices and Nvidia ARM64 architecture devices. 
## Installation
By using anaconda3, it is possible to create different and isolates (like sand boxes) python environments.
During the creation phase it is also pissible to select the version of the interpreter.   
For this project we'll create Python 3.8 v-env called *'GPKF'* by using a .yaml config file:  

### Create the environment
Go into the Installation folder and run the following command to create an empty environment
config file.
```shell
conda create -n GPKF python=3.8
```
Before installing the required python packages, **the environment must be activated**.  
**NOTE:** this action is required whenever you want to work with the desired environment.

### Activate the environment:  

```shell
conda activate GPKF
```
### Install extra conda packages:  
```shell
conda install -c conda-forge empy
```
### Install the packages:  

```shell
python setup.py install
```

**NOTE:** this action must be performed from the installation folder and with the venv active.

## Execution:
The '*experiments*' folder contains two subdirectories that contain the example code to use the GaPt toolkit and to perform 
some benchmarks. Note that each file needs the relative .json config file that can be passed as argument:

```shell
python -m <script_file>.py -c <config_file>.json. 
```
**NOTE:** this action must be performed from the installation folder (Root) and with the venv active.

 
