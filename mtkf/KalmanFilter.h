#pragma once
#include <Eigen/Dense>

template <int StateDim, int MeasDim>
class KalmanFilter {
public:
    KalmanFilter() {
        this->x_ = Eigen::Matrix<double, StateDim, 1>::Zero();
        this->P_ = Eigen::Matrix<double, StateDim, StateDim>::Identity();
        this->I_ = Eigen::Matrix<double, StateDim, StateDim>::Identity();
    };

    void init(const Eigen::Matrix<double, StateDim, 1>& initial_state,
              const Eigen::Matrix<double, StateDim, StateDim>& initial_covariance) {
        this->x_ = initial_state;
        this->P_ = initial_covariance;
    }

    void predict(const Eigen::Matrix<double, StateDim, StateDim>& transition_matrix,
                 const Eigen::Matrix<double, StateDim, StateDim>& process_noise) {
        this->x_ = transition_matrix * this->x_;
        this->P_ = transition_matrix * this->P_ * transition_matrix.transpose() + process_noise;
    }

    void update(const Eigen::Matrix<double, MeasDim, 1>& meas,
            const Eigen::Matrix<double, MeasDim, StateDim>& observation_matrix,
            const Eigen::Matrix<double, MeasDim, MeasDim>& sensor_noise) {
        auto y = meas - observation_matrix * this->x_;
        auto S = observation_matrix * this->P_ * observation_matrix.transpose() + sensor_noise;
        auto K = this->P_ * observation_matrix.transpose() * S.inverse();
        this->x_ += K * y;
        this->P_ = (this->I_ - K * observation_matrix) * this->P_;
    }

    Eigen::Matrix<double, StateDim, 1> const getState() {
        return this->x_;
    }

    Eigen::Matrix<double, StateDim, StateDim> const getCovariance() {
        return this->P_;
    }


private:
    Eigen::Matrix<double, StateDim, 1> x_;          // State
    Eigen::Matrix<double, StateDim, StateDim> P_;   // State covariance
    Eigen::Matrix<double, StateDim, StateDim> I_;   // Identity matrix
};
