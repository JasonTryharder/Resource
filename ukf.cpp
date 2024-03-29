#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3-n_x_;
  is_initialized_ = false;
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  double time_S;
  std::cout << time_S << " 1 " << time_us_ << " 1 " << std::endl;
  if(!is_initialized_)
    {
      x_ <<   5.7441,
              1.3800,
              2.2049,
              0.5015,
              0.3528;
      P_ <<   0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
              -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
              0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
              -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
              -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
      std::cout <<" I am initialized  ???????????????????????????????????????????????????????????????????????????????????????????"<< std::endl;
    is_initialized_ = true;

    }
    
    // Prediction(meas_package.timestamp_/1e8);
    // std::cout <<meas_package.timestamp_ << " The time in us "<< std::endl;
    
    
    
  if (meas_package.sensor_type_ ==0)
  {
    time_S = (meas_package.timestamp_ - time_us_)/1e6; 
    std::cout <<time_S << " The time in seconds "<< std::endl;
    Prediction(time_S);
    UpdateLidar(meas_package);
    time_us_ = meas_package.timestamp_;
    std::cout << time_S << " 2 " << time_us_ << " 2 " << std::endl;
    is_initialized_ = true;

  }
  if (meas_package.sensor_type_ ==1)
  {
    time_S = (meas_package.timestamp_ - time_us_)/1e6;
    Prediction(time_S);
    UpdateLidar(meas_package);
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;// update the deltaT
  }
 
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // std::cout << delta_t<<" I am the time" <<std::endl;
  if (true)
  {
  VectorXd x_aug = VectorXd(7);
  MatrixXd P_aug = MatrixXd(7,7);
  //Create augmented mean and covariance
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;
  MatrixXd Xsig_aug = MatrixXd(n_aug_,2*n_aug_+1);
  MatrixXd A = P_aug;
  A = A.llt().matrixL();
  ////////////////////////////////////////////////////////////////////////////////////////

  Xsig_aug.col(0) = x_aug;
  for (int i = 0;i<n_aug_;++i)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_)= x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }
  Xsig_pred_ = MatrixXd(n_x_,2*n_aug_+1);
  //predict sigma points
  for (int i = 0;i<2*n_aug_ + 1;++i)
  {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    // predicted state values
    double px_p, py_p;
    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  // create vector for weights
  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  // set weights
  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) 
  {  // 2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights(i) = weight;
  }
  // predict state mean and state covariance matrix
  for (int i = 0; i <2 * n_aug_ + 1;++i)
  {
    x_ = x_ + weights(i)*Xsig_pred_.col(i);
  }
  for (int i = 0; i <2 * n_aug_ + 1;++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    P_ = P_ + weights(i) * x_diff * x_diff.transpose();
  }
  }

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    // std::cout <<" where I am running" <<std::endl;
  int n_z1 = 2;
    std::cout<< " I am in Update " << std::endl;

  // create vector for weights
  VectorXd weights1 = VectorXd(2 * n_aug_ + 1);
  // set weights
  double weight_01 = lambda_/(lambda_ + n_aug_);
  weights1(0) = weight_01;
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) 
  {  // 2n+1 weights
    double weight1 = 0.5/(n_aug_+lambda_);
    weights1(i) = weight1;
  }
  // create sigma points in measurement space and predicted measurement mean and covariance 
  MatrixXd Zsig1 = MatrixXd(n_z1, 2 * n_aug_ + 1);
  VectorXd z_pred1 = VectorXd(n_z1);
  MatrixXd S1 = MatrixXd(n_z1,n_z1);
  VectorXd z1 = VectorXd(n_z1);
  z1 = meas_package.raw_measurements_;
  for (int i = 0;i<2*n_aug_+1;++i)
  {
    double p_x1 = Xsig_pred_(0,i);
    double p_y1 = Xsig_pred_(1,i);
    // measurement model
    Zsig1(0,i) = p_x1;                       // px
    Zsig1(1,i) = p_y1;                       // py
  }
  // mean predicted measurement
  z_pred1.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) 
  {
    z_pred1 = z_pred1 + weights1(i) * Zsig1.col(i);
  }
  // innovation covariance matrix S
  S1.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff1 = Zsig1.col(i) - z_pred1;

    // // angle normalization
    // while (z_diff1(1)> M_PI) z_diff1(1)-=2.*M_PI;
    // while (z_diff1(1)<-M_PI) z_diff1(1)+=2.*M_PI;

    S1 = S1 + weights1(i) * z_diff1 * z_diff1.transpose();
  }
  // add measurement noise covariance matrix
  MatrixXd R1 = MatrixXd(n_z1,n_z1);
  R1 <<  std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;
  S1 = S1 + R1;
  MatrixXd Tc1 = MatrixXd(n_x_, n_z1);
  // calculate cross correlation matrix
  Tc1.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  // 2n+1 simga points
    // residual
    VectorXd z_diff1 = Zsig1.col(i) - z_pred1;
    // // angle normalization
    // while (z_diff1(1)> M_PI) z_diff1(1)-=2.*M_PI;
    // while (z_diff1(1)<-M_PI) z_diff1(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff1 = Xsig_pred_.col(i) - x_;
    // // angle normalization
    // while (x_diff1(3)> M_PI) x_diff1(3)-=2.*M_PI;
    // while (x_diff1(3)<-M_PI) x_diff1(3)+=2.*M_PI;

    Tc1 = Tc1 + weights1(i) * x_diff1 * z_diff1.transpose();
  }
  // Kalman gain K;
  MatrixXd K1 = Tc1 * S1.inverse();

  // residual
  VectorXd z_diff11 = z1.col(0) - z_pred1.col(0);// size problem 
  std::cout<< K1.col(0).size() << " X " <<K1.row(0).size()<< "  K1    " << z1.col(0).size() << " X " << z1.row(0).size() << std::endl;
  std::cout<<P_.col(0).size() << " X " << P_.row(0).size() << std::endl;
  // // angle normalization
  // while (z_diff1(1)> M_PI) z_diff1(1)-=2.*M_PI;
  // while (z_diff1(1)<-M_PI) z_diff1(1)+=2.*M_PI;

  // update state mean and covariance matrix
  // x_ = x_ + K1 * z_diff1;
  // P_ = P_ - K1*S1*K1.transpose();




}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // for radar measurement rho, phi, rho_dot
  // int n_z = 3;
  // std::cout<< " I am in Update " << std::endl;
  // // create vector for weights
  // VectorXd weights = VectorXd(2 * n_aug_ + 1);
  // // set weights
  // double weight_0 = lambda_/(lambda_ + n_aug_);
  // weights(0) = weight_0;
  // for (int i = 1; i < 2 * n_aug_ + 1; ++i) 
  // {  // 2n+1 weights
  //   double weight = 0.5/(n_aug_+lambda_);
  //   weights(i) = weight;
  // }
  // // create sigma points in measurement space and predicted measurement mean and covariance 
  // MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  // VectorXd z_pred = VectorXd(n_z);
  // MatrixXd S = MatrixXd(n_z,n_z);
  // VectorXd z = VectorXd(n_z);
  // z = meas_package.raw_measurements_;
  // for (int i = 0;i<2*n_aug_+1;++i)
  // {
  //   double p_x = Xsig_pred_(0,i);
  //   double p_y = Xsig_pred_(1,i);
  //   double v  = Xsig_pred_(2,i);
  //   double yaw = Xsig_pred_(3,i);

  //   double v1 = cos(yaw)*v;
  //   double v2 = sin(yaw)*v;
  //   // measurement model
  //   Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
  //   Zsig(1,i) = atan2(p_y,p_x);                                // phi
  //   Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  // }
  // // mean predicted measurement
  // z_pred.fill(0.0);
  // for (int i=0; i < 2*n_aug_+1; ++i) 
  // {
  //   z_pred = z_pred + weights(i) * Zsig.col(i);
  // }
  // // innovation covariance matrix S
  // S.fill(0.0);
  // for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
  //   // residual
  //   VectorXd z_diff = Zsig.col(i) - z_pred;

  //   // angle normalization
  //   while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  //   while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //   S = S + weights(i) * z_diff * z_diff.transpose();
  // }
  // // add measurement noise covariance matrix
  // MatrixXd R = MatrixXd(n_z,n_z);
  // R <<  std_radr_*std_radr_, 0, 0,
  //       0, std_radphi_*std_radphi_, 0,
  //       0, 0,std_radrd_*std_radrd_;
  // S = S + R;
  // MatrixXd Tc = MatrixXd(n_x_, n_z);
  // // calculate cross correlation matrix
  // Tc.fill(0.0);
  // for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  // {  // 2n+1 simga points
  //   // residual
  //   VectorXd z_diff = Zsig.col(i) - z_pred;
  //   // angle normalization
  //   while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  //   while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //   // state difference
  //   VectorXd x_diff = Xsig_pred_.col(i) - x_;
  //   // angle normalization
  //   while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
  //   while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

  //   Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  // }
  // // Kalman gain K;
  // MatrixXd K = Tc * S.inverse();

  // // residual
  // VectorXd z_diff = z - z_pred;

  // // angle normalization
  // while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  // while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // // update state mean and covariance matrix
  // x_ = x_ + K * z_diff;
  // P_ = P_ - K*S*K.transpose();



}