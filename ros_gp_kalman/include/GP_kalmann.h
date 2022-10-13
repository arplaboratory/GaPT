#pragma once
#include <math.h>
#include <Eigen/Eigen>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

struct state
{
	Eigen::MatrixXd Covar;
	Eigen::VectorXd Mean;
};


// THE IDEA OF THIS CLASS. This class is meant to model SISO process which gets continuous feedback on its data.
// predict(t) gives you a guess of the process value at value t
// the closer t is to curr_t in the state the better your prediction
// update(zt) is the measurement update or the real value. The more frequently this is called with more data the better this process becomes.

class GP_kalmann
{
private:
	//Order of your approximation to the GP, your state space will be a vector Order X 1 Large.
	//Also a covraince matrix Order x Order big. 
	int order;
	//Problem we assume a continuous state equation. t is your input variable. It doesn't need to be time.
	//dx/dt = Fx(t) + Lu
	//Z = Hx;
	Eigen::MatrixXd F;
	//This holds the Model noise
	Eigen::MatrixXd Qv;
	//This is the Sensor Update matrix z = Cx;
	Eigen::MatrixXd H; 
	//This is the noise update or B matrix 
	Eigen::MatrixXd L;
	//This is your sensor noise.
	double R; 
    //This is your covar diffusion matrix
    Eigen::MatrixXd Q;

	// This calculated e^(F*dt). This is mostly called by the predict function 
	Eigen::MatrixXd expm(double dt);
	//This calculates the ideal Noise update for the covariance. Mostly called by predict function
	Eigen::MatrixXd calc_Qs(double dt, Eigen::MatrixXd A);
	state curr_state; //The internal current state of the Kalman Filter
	double curr_t; //The Current input that you are approximated around
	std::vector<double>  x_data;// = drag_list["xtrain"];
	std::vector<double>  y_data;// = drag_list["ytrain"];
	int binarySearch(int l, int r, double x);

public:
	//Constructor this string should be the path to your node handle parameter or launch file definition given.
	GP_kalmann(std::string path_yaml);
	//Manually set all the Matrixes for you
	GP_kalmann(Eigen::MatrixXd Fin, Eigen::MatrixXd Hin, Eigen::MatrixXd Lin, double Qin,double R);
	//Make a prediction step by moving the kalman filter if your GP~Norm_dist(g(curr_t),covar(curr_t))
	//This would return a N(g(t+dt),covar(t+dt))
	state predict(double t);
	//Updates the state based on the measurement. This is equiviliant to redoing the approximation. 
	state update(double zt);
	//Set the current Sensor Noise R
	void setSenNoise(double x);
	//Set the current state
	void setState(state input, double t);
	state getState();
	double getCurr_t();

	void loadDataset(std::vector<double>  x_data_in, std::vector<double>  y_data_in);
	state gp_regress(double in);
};

