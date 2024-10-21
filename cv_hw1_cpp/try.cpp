#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <cstring>

int main()
{
    cv::Mat a(3,3,CV_16U,cv::Scalar(3));
    a.at<uint16_t>(2,1) = 2;
    std::cout << a << std::endl;
    std::vector<int> c={1,2,3,4,5};
    c.resize(10);
    std::cout << c[7];
}