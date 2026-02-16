find_package(Eigen3 REQUIRED)

# Library target
add_library(libmtkf MTKF.cpp)
target_include_directories(libmtkf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libmtkf PUBLIC Eigen3::Eigen)

