// C++ KalmannFilter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <GP_kalmann.h>
#include <ros/ros.h>
#include <gnuplot.h>
#include <fstream>
using namespace std;
int main(int argc, char** argv)
{
	ros::init(argc, argv, "GP_Kalmann example");
	ros::NodeHandle nh;
        ros::start();
	std::string axis;
	nh.getParam("/gp_kalmann/axis", axis);
	std::string drag_item = "/gp_kalmann/"+axis;
	int order;
	nh.getParam(drag_item+"/Order", order);
	std::cout << order <<std::endl;
	GP_kalmann filter(drag_item);
        Eigen::VectorXd X = Eigen::VectorXd::Zero(order);
        Eigen::MatrixXd Covar = Eigen::MatrixXd::Identity(order, order);
	std::vector<double> Pinf;
	nh.getParam(drag_item+"/P_inf", Pinf);
	int count = 0;
	for (int i = 0;i<order;i++){
		for (int j = 0;j<order;j++){
			Covar(i,j) = Pinf[count];
			count +=1;
		}
	}
	std::vector<double>  x_data;// = drag_list["xtrain"];
	std::vector<double>  y_data;// = drag_list["ytrain"];
	nh.getParam(drag_item+"/xtrain", x_data);	
	nh.getParam(drag_item+"/ytrain", y_data);	
	std::cout << "dataset size" << x_data.size() <<std::endl;
	filter.loadDataset(x_data,y_data);
	int mid_point =1;//floor(x_data.size()/2);
        state currin;
        currin.Mean = X;
        currin.Covar = Covar;
        filter.setState(currin,x_data[mid_point]);
        ofstream outFileEst;
        ofstream outFileTrue;
        outFileEst.open("GP_est.dat");
        outFileTrue.open("GP_True.dat");
        count = 1;
	state curr = currin;
	for(int i=mid_point;i<x_data.size();i+=1){
		outFileEst<< x_data[i];
		ros::Time lasttime=ros::Time::now();
		curr = filter.gp_regress(x_data[i]+0.02);
		ros::Time currtime=ros::Time::now();
		ros::Duration diff=currtime-lasttime;
		std::cout<<"diff: "<<diff<<endl;
		outFileEst << " " <<curr.Mean(0)<< endl;
		curr = filter.update(y_data[i]);
		outFileTrue<< x_data[i];
		outFileTrue << " " <<y_data[i]<< endl;
\
	}
        GnuplotPipe gp;
        gp.sendLine("plot 'GP_True.dat' , 'GP_est.dat'  ");
	outFileEst.close();
	outFileTrue.close();
}
