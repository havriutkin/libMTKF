cmake_minimum_required(VERSION 3.10)
project(libMTKF CXX)
set(CMAKE_CXX_STANDARD 17)

# Build library and subprojects
# library sources live in the `mtkf` subdirectory now
add_subdirectory(mtkf)
add_subdirectory(examples)
add_subdirectory(test)


