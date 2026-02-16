#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "mtkf/KalmanFilter.h"

// Simple test for a 2D state, 1D measurement Kalman Filter
TEST(KalmanFilterTest, PredictAndUpdate) {
    constexpr int StateDim = 2;
    constexpr int MeasDim = 1;
    KalmanFilter<StateDim, MeasDim> kf;

    // Initial state and covariance
    Eigen::Matrix<double, StateDim, 1> x0;
    x0 << 0, 0;
    Eigen::Matrix<double, StateDim, StateDim> P0 = Eigen::Matrix<double, StateDim, StateDim>::Identity();
    kf.init(x0, P0);

    // Transition matrix (identity)
    Eigen::Matrix<double, StateDim, StateDim> F = Eigen::Matrix<double, StateDim, StateDim>::Identity();
    // Process noise (small)
    Eigen::Matrix<double, StateDim, StateDim> Q = 0.01 * Eigen::Matrix<double, StateDim, StateDim>::Identity();

    // Predict
    kf.predict(F, Q);

    // Observation matrix (observe first state only)
    Eigen::Matrix<double, MeasDim, StateDim> H;
    H << 1, 0;
    // Sensor noise (small)
    Eigen::Matrix<double, MeasDim, MeasDim> R;
    R << 0.01;
    // Measurement (observe 1)
    Eigen::Matrix<double, MeasDim, 1> z;
    z << 1;

    // Update
    kf.update(z, H, R);

    // After update, first state should move toward 1
    // Second state should remain close to 0
    // Covariance should decrease
    auto x = kf.getState();
    auto P = kf.getCovariance();
    // First state should be positive and less than 1
    EXPECT_GT(x(0), 0.0);
    EXPECT_LT(x(0), 1.0);
    // Second state should be close to 0
    EXPECT_NEAR(x(1), 0.0, 1e-6);
    // Covariance for observed state should be less than initial
    EXPECT_LT(P(0,0), 1.0);
    // Covariance for unobserved state may increase due to process noise, so no check here
}
