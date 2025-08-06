cmake_minimum_required(VERSION 3.10)   # 指定最低CMake版本

project(Project)  # 项目名和版本，支持的语言

set(CMAKE_CXX_STANDARD 17)      # 使用C++17标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 必须使用指定标准

find_package(Eigen3 REQUIRED)

add_executable(DLT DLT.cpp)
target_link_libraries(DLT PRIVATE Eigen3::Eigen)

add_executable(GN GN.cpp
        data.h)
target_link_libraries(GN PRIVATE Eigen3::Eigen)

add_executable(GD GD.cpp
        data.h)
target_link_libraries(GD PRIVATE Eigen3::Eigen)

add_executable(LM LM.cpp
        data.h)
target_link_libraries(LM PRIVATE Eigen3::Eigen)
