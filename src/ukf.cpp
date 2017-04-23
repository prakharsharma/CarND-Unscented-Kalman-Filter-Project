#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // dimensions of the state vector
  n_x_ = 5;

  // dimensions of the augmented sigma points
  n_aug_ = n_x_ + 2;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // initial predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

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
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;

  previous_timestamp_ = 0;

  lambda_ = 3 - n_aug_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_) {
    ProcessFirstMeasurement(meas_package);
  } else {
    ProcessSubsequentMeasurement(meas_package);
  }

  previous_timestamp_ = meas_package.timestamp_;
}

void UKF::ProcessFirstMeasurement(MeasurementPackage meas_package) {

  double px = 0.0, py = 0.0, v = 0.0, yaw = 0.0, yaw_rate = 0.0;

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    px = meas_package.raw_measurements_[0];
    py = meas_package.raw_measurements_[1];

    // TODO: initialize v, yaw and yaw_rate in case of Laser measurement
  } else {
    double ro = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];
    px = ro * cos(phi);
    py = ro * sin(phi);

    // TODO: initialize v, yaw and yaw_rate in case of Radar measurement
  }
  x_ << px, py, v, yaw, yaw_rate;
  is_initialized_ = true;
}

void UKF::ProcessSubsequentMeasurement(MeasurementPackage meas_package) {

  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  // 1. Generate augmented sigma points
  MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(Xsig_aug);

  // 2. Predict sigma points
  SigmaPointPrediction(Xsig_aug, delta_t);

  // 3. Predict mean and covariance
  PredictMeanAndCovariance();

}

void UKF::AugmentedSigmaPoints(MatrixXd& Xsig_aug) {

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  // create augmented state covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  //calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  Xsig_aug.col(0)  = x_aug;

  for (uint i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }
}

void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, double delta_t) {

  Xsig_pred_.fill(0.0);

  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {

    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yaw_rate = Xsig_aug(4, i);
    double noise_a = Xsig_aug(5, i);
    double noise_yaw_rate = Xsig_aug(6, i);

    if (fabs(yaw_rate) < 0.00001) {
      px += v * cos(yaw) * delta_t;
      py += v * sin(yaw) * delta_t;
    } else {
      px += v/yaw_rate * (sin(yaw + yaw_rate * delta_t) - sin(yaw));
      py += v/yaw_rate * (cos(yaw) - cos(yaw + yaw_rate * delta_t));
    }
    px += 0.5 * delta_t * delta_t * cos(yaw) * noise_a;
    py += 0.5 * delta_t * delta_t * sin(yaw) * noise_a;

    v += delta_t * noise_a;

    yaw += yaw_rate * delta_t;
    yaw += 0.5 * delta_t * delta_t * noise_yaw_rate;

    yaw_rate += delta_t * noise_yaw_rate;

    Xsig_pred_(0, i) = px;
    Xsig_pred_(1, i) = py;
    Xsig_pred_(2, i) = v;
    Xsig_pred_(3, i) = yaw;
    Xsig_pred_(4, i) = yaw_rate;
  }

}

void UKF::PredictMeanAndCovariance() {

  int n_sig = 2 * n_aug_ + 1;

  // create vector for weights
  VectorXd weights(n_sig);

  weights(0) = lambda_/(lambda_ + n_aug_);
  for (uint i = 1; i < n_sig; i++) {
    weights(i) = 0.5/(lambda_ + n_aug_);
  }

  // predict state mean
  x_.fill(0.0);
  for (uint i = 0; i < n_sig; i++) {
    x_ += weights(i) * Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P_.fill(0.0);
  for (uint i = 0; i < n_sig; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3) > M_PI) {
      x_diff(3) -= 2.0 * M_PI;
    }

    while (x_diff(3) < -M_PI) {
      x_diff(3) += 2.0 * M_PI;
    }

    P_ = P_ + weights(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
