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
  x_ <<   5.7441,
      1.3800,
      2.2049,
      0.5015,
      0.3528;

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
//  P_ <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
//      -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
//      0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
//      -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
//      -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
  P_.fill(0.0);

  // initial predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

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

  weights_ = VectorXd(2 * n_aug_ + 1);

  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (uint i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5/(lambda_ + n_aug_);
  }

  // set measurement dimension, radar can measure r, phi, and r_dot
  n_radz_ = 3;

  // initialize radar measurement noise covariance matrix
  R_rad_ = MatrixXd(n_radz_, n_radz_);
  R_rad_.fill(0.0);
  R_rad_(0, 0) = std_radr_ * std_radr_;
  R_rad_(1, 1) = std_radphi_ * std_radphi_;
  R_rad_(2, 2) = std_radrd_ * std_radrd_;

  // laser can measure px, py
  n_lasz_ = 2;

  // initialize laser measurement noise covariance matrix
  R_las_ = MatrixXd(n_lasz_, n_lasz_);
  R_las_.fill(0.0);
  R_las_(0, 0) = std_laspx_ * std_laspx_;
  R_las_(1, 1) = std_laspy_ * std_laspy_;

  n_total_readings_laser_ = 0;
  n_total_readings_radar_ = 0;
  n_processed_readings_laser_ = 0;
  n_processed_readings_radar_ = 0;

  // Initialize radar NIS buckets
  nis_buckets_radar_["0.95"] = pair<double, uint>(0.352, 0);
  nis_buckets_radar_["0.90"] = pair<double, uint>(0.584, 0);
  nis_buckets_radar_["0.10"] = pair<double, uint>(6.251, 0);
  nis_buckets_radar_["0.05"] = pair<double, uint>(7.815, 0);

  // Initialize laser NIS buckets
  nis_buckets_laser_["0.95"] = pair<double, uint>(0.103, 0);
  nis_buckets_laser_["0.90"] = pair<double, uint>(0.211, 0);
  nis_buckets_laser_["0.10"] = pair<double, uint>(4.605, 0);
  nis_buckets_laser_["0.05"] = pair<double, uint>(5.991, 0);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_) {
    ProcessFirstMeasurement(meas_package);
    is_initialized_ = true;
  } else {
    ProcessSubsequentMeasurement(meas_package);
  }

  previous_timestamp_ = meas_package.timestamp_;
}

void UKF::ProcessFirstMeasurement(MeasurementPackage meas_package) {

  double px = 0.0, py = 0.0, v = 0.0, yaw = 0.0, yaw_rate = 0.0;

  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    n_processed_readings_laser_++;

    px = meas_package.raw_measurements_[0];
    py = meas_package.raw_measurements_[1];

    // TODO: initialize v, yaw and yaw_rate in case of Laser measurement
  } else if (use_radar_ && meas_package.sensor_type_ ==
      MeasurementPackage::RADAR) {
    n_processed_readings_radar_++;

    double ro = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];
    px = ro * cos(phi);
    py = ro * sin(phi);

    // TODO: initialize v, yaw and yaw_rate in case of Radar measurement
  }
  x_ << px, py, v, yaw, yaw_rate;

  // TODO: correctly initialize covariance matrix
  P_.fill(0.0);
}

void UKF::ProcessSubsequentMeasurement(MeasurementPackage meas_package) {

  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;

  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    n_processed_readings_radar_++;
    Prediction(delta_t);
    UpdateRadar(meas_package);
  } else if (use_laser_ && meas_package.sensor_type_ ==
      MeasurementPackage::LASER) {
    n_processed_readings_laser_++;
    Prediction(delta_t);
    UpdateLidar(meas_package);
  }

//  cout
//      << "################################################"
//      << " Processed "
//      << n_processed_readings_radar_ << "/" << n_total_readings_radar_
//      << " radar readings and "
//      << n_processed_readings_laser_ << "/" << n_total_readings_laser_
//      << " laser readings "
//      << "################################################"
//      << endl;
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
//  cout << "Augemented sigma points" << endl << Xsig_aug << endl;

  // 2. Predict sigma points
  SigmaPointPrediction(Xsig_aug, delta_t);
//  cout << "Predicted sigma points" << endl << Xsig_pred_ << endl;

  // 3. Predict mean and covariance
  PredictMeanAndCovariance();
//  cout << "Predicted x_ " << endl << x_ << endl;
//  cout << "Predicted P_ " << endl << P_ << endl;

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

  // predict state mean
  x_.fill(0.0);
  for (uint i = 0; i < n_sig; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
//  cout << "Predicted x_\n" << x_ << endl;

  //predict state covariance matrix
  P_.fill(0.0);
  for (uint i = 0; i < n_sig; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    while (x_diff(3) > M_PI) {
      double fac = floor(fabs(x_diff(3)) / M_PI) * M_PI;
      x_diff(3) -= fac;
    }
    while (x_diff(3) < -M_PI) {
      double fac = floor(fabs(x_diff(3)) / M_PI) * M_PI;
      x_diff(3) += fac;
    }

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
//  cout << "Predicted P_\n" << P_ << endl;

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

  // 1. Predict laser measurement and covariance matrix

  // create matrix for sigma points in measurement space
  MatrixXd Zsig(n_lasz_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred(n_lasz_);

  // measurement covariance matrix S
  MatrixXd S(n_lasz_, n_lasz_);

  PredictLaserMeasurement(Zsig, z_pred, S);
//  cout
//      << "Predicted laser measurements" << endl
//      << "Zsig\n" << Zsig << endl
//      << "z_pred\n" << z_pred << endl
//      << "S\n" << S << endl;

  // 2. Update belief about object's position
  UpdateStateUsingLaserMeasurement(meas_package, Zsig, z_pred, S);
//  cout
//      << "Updated measurements" << endl
//      << "x_\n" << x_ << endl
//      << "P_\n" << P_ << endl
//      <<"NIS_laser_ " << NIS_laser_ << endl;

  for (auto it = nis_buckets_laser_.begin();
       it != nis_buckets_laser_.end(); it++) {
    string key = it->first;
    double threshold = it->second.first;
    uint counter = it->second.second;
    if (NIS_laser_ >= threshold) {
      counter++;
    }
    nis_buckets_laser_[key] = pair<double, uint>(threshold, counter);
  }
}

void UKF::PredictLaserMeasurement(MatrixXd& Zsig, VectorXd& z_pred,
                                  MatrixXd& S) {

  // transform sigma points into measurement space
  Zsig.fill(0.0);
  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);

    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate measurement covariance matrix S
  S.fill(0.0);
  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {
    // measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R_las_;
}

void UKF::UpdateStateUsingLaserMeasurement(
    const MeasurementPackage& meas_package,
    const MatrixXd& Zsig,
    const VectorXd& z_pred,
    const MatrixXd& S) {

  //create matrix for cross correlation Tc
  MatrixXd Tc(n_x_, n_lasz_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd S_inv = S.inverse();

  // calculate Kalman gain K;
  MatrixXd K = Tc * S_inv;

  // update state
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  x_ = x_ + K * z_diff;

  // update covariance matrix
  P_ -= K * S * K.transpose();

  // calculate NIS
  NIS_laser_ = z_diff.transpose() * S_inv * z_diff;
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

  // 1. Predict radar measurement and covariance matrix

  // create matrix for sigma points in measurement space
  MatrixXd Zsig(n_radz_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred(n_radz_);

  // measurement covariance matrix S
  MatrixXd S(n_radz_, n_radz_);

  PredictRadarMeasurement(Zsig, z_pred, S);
//  cout
//      << "Predicted radar measurements" << endl
//      << "Zsig\n" << Zsig << endl
//      << "z_pred\n" << z_pred << endl
//      << "S\n" << S << endl;

  // 2. Update belief about object's position
  UpdateStateUsingRadarMeasurement(meas_package, Zsig, z_pred, S);
//  cout
//      << "Updated measurements" << endl
//      << "x_\n" << x_ << endl
//      << "P_\n" << P_ << endl
//      <<"NIS_radar_ " << NIS_radar_ << endl;

  for (auto it = nis_buckets_radar_.begin();
       it != nis_buckets_radar_.end(); it++) {
    string key = it->first;
    double threshold = it->second.first;
    uint counter = it->second.second;
    if (NIS_radar_ >= threshold) {
      counter++;
    }
    nis_buckets_radar_[key] = pair<double, uint>(threshold, counter);
  }

}

void UKF::PredictRadarMeasurement(MatrixXd& Zsig, VectorXd& z_pred,
                                  MatrixXd& S) {

  // transform sigma points into measurement space
  Zsig.fill(0.0);
  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {

    // transform one sigma point into measurement space
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double rho = sqrt(px * px + py * py);
    if (rho < 0.00001) {
      // avoid division by zero
      continue;
    }

    double phi = atan2(py, px);
    while (phi > M_PI) {
      phi -= 2.0 * M_PI;
    }
    while (phi < -M_PI) {
      phi += 2.0 * M_PI;
    }

    double rho_dot = (px * cos(yaw) * v + py * sin(yaw) * v)/rho;

    Zsig(0, i) = rho;
    Zsig(1, i) = phi;
    Zsig(2, i) = rho_dot;
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate measurement covariance matrix S
  S.fill(0.0);
  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {
    // measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) {
//      z_diff(1) -= 2.0 * M_PI;
      z_diff(1) -= floor(fabs(z_diff(1)) / M_PI) * M_PI;
    }
    while (z_diff(1) < -M_PI) {
//      z_diff(1) += 2.0 * M_PI;
      z_diff(1) += floor(fabs(z_diff(1)) / M_PI) * M_PI;
    }

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R_rad_;
}

void UKF::UpdateStateUsingRadarMeasurement(
    const MeasurementPackage& meas_package,
    const MatrixXd& Zsig,
    const VectorXd& z_pred,
    const MatrixXd& S) {

  //create matrix for cross correlation Tc
  MatrixXd Tc(n_x_, n_radz_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (uint i = 0; i < 2 * n_aug_ + 1; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) > M_PI) {
//      x_diff(3) -= 2.0 * M_PI;
      x_diff(3) -= floor(fabs(x_diff(3)) / M_PI) * M_PI;
    }
    while (x_diff(3) < -M_PI) {
//      x_diff(3) += 2.0 * M_PI;
      x_diff(3) += floor(fabs(x_diff(3)) / M_PI) * M_PI;
    }

    // measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI) {
//      z_diff(1) -= 2.0 * M_PI;
      z_diff(1) -= floor(fabs(z_diff(1)) / M_PI) * M_PI;
    }
    while (z_diff(1) < -M_PI) {
//      z_diff(1) += 2.0 * M_PI;
      z_diff(1) += floor(fabs(z_diff(1)) / M_PI) * M_PI;
    }

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd S_inv = S.inverse();

  // calculate Kalman gain K;
  MatrixXd K = Tc * S_inv;

  // update state
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  while (z_diff(1) > M_PI) {
    z_diff(1) -= 2.0 * M_PI;
  }
  while (z_diff(1) < -M_PI) {
    z_diff(1) += 2.0 * M_PI;
  }
  x_ = x_ + K * z_diff;

  // update covariance matrix
  P_ -= K * S * K.transpose();

  // calculate NIS
  NIS_radar_ = z_diff.transpose() * S_inv * z_diff;

}
