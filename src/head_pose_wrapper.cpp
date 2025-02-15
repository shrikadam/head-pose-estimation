#include "head_pose_wrapper.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

py::module_ head_pose_module;

void initialize_python() {
    Py_Initialize();
    
    // Add the current directory to Python's module search path
    PyRun_SimpleString("import sys; sys.path.append('.')");
    
    // Import our Python module
    head_pose_module = py::module_::import("head_pose_module");
}

cv::Mat process_frame_cpp(const cv::Mat& frame) {
    // Convert OpenCV Mat to numpy array
    py::array_t<unsigned char> py_image({frame.rows, frame.cols, 3}, frame.data, py::cast(frame));

    // Call Python function
    py::object result = head_pose_module.attr("process_frame")(py_image);

    // Convert result back to OpenCV Mat
    py::buffer_info buf = result.cast<py::array_t<unsigned char>>().request();
    cv::Mat processed_frame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    return processed_frame.clone();  // Return a copy to ensure memory safety
}

cv::Mat detect_head_pose(const cv::Mat& frame) {
    // Convert OpenCV Mat to numpy array
    py::array_t<unsigned char> py_image({frame.rows, frame.cols, 3}, frame.data, py::cast(frame));

    // Call Python function
    py::object result = head_pose_module.attr("process_frame")(py_image);

    // Convert result back to OpenCV Mat
    py::buffer_info buf = result.cast<py::array_t<unsigned char>>().request();
    cv::Mat processed_frame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    return processed_frame.clone();  // Return a copy to ensure memory safety
}

void finalize_python() {
    Py_Finalize();
}