cmake_minimum_required(VERSION 3.12)
project(HeadPoseEstimator)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add your source files
add_executable(${PROJECT_NAME} src/track_head.cpp src/head_pose_wrapper.cpp)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    src
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Python3::Python
    Python3::NumPy
    pybind11::embed
)

# Copy Python script to build directory
configure_file(src/head_pose_module.py ${CMAKE_BINARY_DIR}/head_pose_module.py COPYONLY)

# If you have a model file, copy it as well
configure_file(src/model.pkl ${CMAKE_BINARY_DIR}/model.pkl COPYONLY)
