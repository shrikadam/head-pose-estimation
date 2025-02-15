#ifndef HEAD_POSE_WRAPPER_H
#define HEAD_POSE_WRAPPER_H

#include <opencv2/opencv.hpp>

void initialize_python();
cv::Mat process_frame_cpp(const cv::Mat& frame);
cv::Mat detect_head_pose(const cv::Mat& frame);
void finalize_python();

#endif // HEAD_POSE_WRAPPER_H