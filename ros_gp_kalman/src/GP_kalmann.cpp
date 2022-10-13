#include <GP_kalmann.h>
#include <iostream>
#include <ros/ros.h>


GP_kalmann::GP_kalmann(std::string path_yaml) {
	ros::NodeHandle nh;
	std::vector<double> Qin;
	std::vector<double> Fin;
	std::vector<double> Lin;
	std::vector<double> Hin;
	nh.getParam(path_yaml+"/Order", order);
	nh.getParam(path_yaml+"/F", Fin);
	nh.getParam(path_yaml+"/L", Lin);
	nh.getParam(path_yaml+"/H", Hin);
	nh.getParam(path_yaml+"/Q", Qin);
	nh.getParam(path_yaml+"/R", R);

	F = Eigen::MatrixXd::Zero(order, order);
	H = Eigen::MatrixXd::Zero(1, order );
	L = Eigen::MatrixXd::Zero(order, order);
        Q = Eigen::MatrixXd::Zero(order, order);

	int count = 0;
	for (int i = 0;i<order;i++){
		for (int j = 0;j<order;j++){
			F(i,j) = Fin[count];
            Q(i,j) = Qin[count];
            L(i,j) = Lin[count];
			count +=1;
        }
		H(0,i) = Hin[i];
		L(i,0) = Lin[i];
	}

    std::cout << "Order: " << order <<std::endl;
	std::cout << "F " << std::endl << F <<std::endl;
	std::cout << "H " << std::endl << H <<std::endl;
	std::cout << "L " << std::endl << L <<std::endl;
	std::cout << "Q " << Q <<std::endl;
	std::cout << "R " << R <<std::endl;

}

GP_kalmann::GP_kalmann(Eigen::MatrixXd Fin, Eigen::MatrixXd Hin, Eigen::MatrixXd Lin, double Qin, double Rin) {
	order = Fin.rows();
	F = Fin;
	Qv = Eigen::MatrixXd::Zero(order * 2, order * 2);
	Qv.block( 0, 0, order, order) = Fin;
	Qv.block(0, order ,order, order) = Lin * Qin * Lin.transpose();
	Qv.block(order, order ,order, order) = -1*Fin.transpose();
	H = Hin;
	L = Lin;
	R = Rin;
}

Eigen::MatrixXd GP_kalmann::expm(double dt) {
	Eigen::MatrixXd Abexp = (F * dt).exp();
	//Eigen::MatrixXd Abexp = Eigen::MatrixXd::Identity(order, order)+F*dt
	return Abexp;
}

//Sets the Sensor noise
void GP_kalmann::setSenNoise(double x){
	R=x;
}

Eigen::MatrixXd GP_kalmann::calc_Qs(double dt, Eigen::MatrixXd At) {
    Qv = Eigen::MatrixXd::Zero(order * 2, order * 2);
    Qv.block(0,0, order, order) = F;
    Qv.block(order, order, order, order)=-F.transpose();
    Qv.block(0,order, order, order) = (L.transpose()*L) * Q;
	Eigen::MatrixXd EigValScales = (Qv * dt).exp();
	return EigValScales.block(0, order, order, order) * expm(dt).transpose();

}



//State is updated based on movement on X
state GP_kalmann::predict(double t) {
	double dt = t- curr_t;
	state new_state;
	Eigen::MatrixXd A = expm(dt);
	new_state.Mean = A * curr_state.Mean;
    new_state.Covar = A * curr_state.Covar * A.transpose()+ calc_Qs(dt, A);
	curr_state = new_state;
	curr_t =t;
	return curr_state;
}


//Updates the state based on the measurement
state GP_kalmann::update(double zt) {
	state upd_state;
	double factor = (H * curr_state.Covar * H.transpose())(0) + R;
	Eigen::MatrixXd Kt = curr_state.Covar * H.transpose() / factor;
	upd_state.Covar = curr_state.Covar - Kt *Kt.transpose()*factor ;
	upd_state.Mean = curr_state.Mean + Kt * (zt- curr_state.Mean(0));
	curr_state = upd_state;
	return curr_state;
}

void GP_kalmann::setState(state input, double t){
	curr_state = input;
	curr_t = t;
}

state GP_kalmann::getState(){
	return curr_state;
}

double GP_kalmann::getCurr_t(){
	return curr_t;
}

int GP_kalmann::binarySearch(int l, int r, double x)
{

    if (r >= l) {
        int mid = l + (r - l) / 2;
 
        // If the element is present at the middle
        // itself
    	if(x_data[mid] < x){
    	     if(x_data[mid+1] > x){
    	     	return mid;
	    }
    	}	
 	if(x_data[mid]==x){
 	     return mid;
	}
        // If element is smaller than mid, then
        // it can only be present in left subarray
        if (x_data[mid] > x)
            return binarySearch( l, mid - 1, x);
 
        // Else the element can only be present
        // in right subarray
        return binarySearch(mid + 1, r, x);
    }
 
    // We reach here when element is not
    // present in array
    return -1;
}
 
state GP_kalmann::gp_regress(double in){
    double dt;
    int count;
    state currin;
    currin.Mean = Eigen::VectorXd::Zero(order);
    currin.Covar =Eigen::MatrixXd::Identity(order, order);
	if(x_data.size()==0){
		std::cout << " No dataset" <<std::endl;
		return currin;
	}
	if(x_data[0]>in){
		currin.Mean(0) = y_data[0];
		return currin;
	}
	if(x_data[x_data.size()-1] < in){
	     count = x_data.size()-1;
	}
	else{
	    count = binarySearch(0, x_data.size()-1, in);
	}
	count-=10;
	if(count <0){
		count =0;
	}
	currin.Mean(0) = y_data[count];
	//First point
	setState(currin,x_data[count]);
	//Iterate through all values
	while(x_data[count]<in && count < x_data.size()){
		predict(x_data[count]);
		update(y_data[count]);
		count+=1;
     	}
    return predict(in);
}


void GP_kalmann::loadDataset(std::vector<double>  x_data_in,std::vector<double>  y_data_in){
	x_data = x_data_in;
	y_data = y_data_in;
}
