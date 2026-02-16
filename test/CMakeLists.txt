enable_testing()

add_executable(test_kalmanfilter test_kalmanfilter.cpp)
# GoogleTest for KalmanFilter
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
target_link_libraries(test_kalmanfilter PRIVATE libmtkf GTest::gtest GTest::gtest_main Eigen3::Eigen)
add_test(NAME kalmanfilter COMMAND test_kalmanfilter)

# Add the project root as an include directory for test_kalmanfilter
target_include_directories(test_kalmanfilter PRIVATE ${CMAKE_SOURCE_DIR})
