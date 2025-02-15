#include <opencv2/opencv.hpp>
#include "head_pose_wrapper.h"

int main() {
    initialize_python();

    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        cv::Mat processed_frame = process_frame_cpp(frame);

        cv::imshow("Head Pose Estimation", processed_frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    finalize_python();

    return 0;
}