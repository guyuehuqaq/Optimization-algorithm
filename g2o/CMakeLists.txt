cmake_minimum_required(VERSION 3.10)   # 指定最低CMake版本
cmake_policy(SET CMP0074 NEW)

project(Project)  # 项目名和版本，支持的语言

set(CMAKE_CXX_STANDARD 17)      # 使用C++17标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 必须使用指定标准

find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)

add_executable(curve_fit_g2o curve_fit_g2o.cpp
        curve_fit_g2o.h)
target_link_libraries(curve_fit_g2o PRIVATE Eigen3::Eigen ${OpenCV_LIBS} g2o::core)
target_include_directories(curve_fit_g2o PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})