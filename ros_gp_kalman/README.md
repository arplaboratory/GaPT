# Gapt C++ Filtering

This repository details the GP Kalmann Filter. Currently, it has no inputs and simply outputs a Gaussian Process Test of the drag data in test_demo.launch. To change the parameters we simply change the model_params.yaml in the launch foulder

Please note: This State Space version assume this is a Single input Single output Gaussian Process. It is Much More Complex to convert a  multidimensional Gaussian Process to a Kalmann Filter. 

# Installation
  
```
$ sudo apt install gnuplot
$ cd your_catkin_workspace/src
$ git pull https://github.com/arplaboratory/gp_kalmann.git
$ catkin build gp_kalmann 
$ cd ..
$ source devel/setup.bash
```

# Demo 

Visaulizes the output of the GP Kalmann Filter. Expected Output should be a graph continaing the dataset of the Drag and the predicted value. Purple estimated value and Green is the True values of the dataset. 
```
$ roslaunch gp_kalmann test.launch
```

Hyperparameters created from this repository. Which is a clone of another paper. https://github.com/arplaboratory/python_gp_kalman_hyperparam

# Launch File Parameters
For now, we assume you have solved the Linear Model Approximation of the GP based on the following slides http://gpss.cc/gpss13/assets/Sheffield-GPSS2013-Sarkka.pdf
This means you have a Kalmann Filter of parameters. Where X is a vector consiting of your output value and its higher order derivatives in the following state. X_dot is the the time derivative of X. Don't worry this model takes care of handling a contiuous time model in the back.

X_dot = Fx+Lu  

Z = Hx

axis - "x_drag", "y_drag", or "z_drag" for the appropiate axis you want to model and plot. 

The format consists of x_drag,y_drag, and z_drag for each of those axises.
- Order The size of your state space Model The 0th index is the value you want. 1st replresents the first derivative etc....\
- Q Kalmann Filter Model Noise
- P_inf Initial Covariance of your State Space Gaussian Process Estimate.
- R Sensor Noise.
- xtest - the inputs to your dataset.
- ytest - the outputs to your dataset.
- X_fact -The amount we scaled by xtest by from the true dataset (true_dataset_inputs)/x_fact = xtest
- (true_dataset_output)*std+meanY = ytest 
- meanY - the Mean of the original dataset that we removed
- std - the range of the original dataset we remvoed. 


