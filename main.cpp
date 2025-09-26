#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace filesystem;

#define HIST_SIZE 256


Mat calculate_hist(const Mat& src_img, int hist_size) {
    // Compute histogram
    float range[] = {0, 256}; // Range of pixel values
    const float* histRange = {range};
    Mat hist;
    calcHist(&src_img, 1, 0, Mat(), hist, 1, &hist_size, &histRange, true, false);

    return hist;
}


// Function to compute and display histogram
void compute_and_display_histogram(const Mat& src_img, const string& windowName, int hist_size) {
    // Compute histogram
    Mat hist = calculate_hist(src_img, hist_size);

    // Create image for histogram display
    int hist_w = 800, hist_h = 600;
    int bin_w = cvRound((double)hist_w / hist_size);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

    // Normalize histogram to fit in image height
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Draw histogram bars
    for (int i = 1; i < hist_size; i++) {
        line(histImage,
             Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
             Scalar(0, 0, 255), 2, 8, 0);
    }

    // Display histogram
    imwrite(windowName, histImage);

    // Print some histogram statistics
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(hist, &minVal, &maxVal, &minLoc, &maxLoc);

    cout << "Histogram Statistics for " << windowName << ":" << endl;
    cout << "Max frequency: " << maxVal << " at intensity: " << maxLoc.y << endl;
    cout << "Min frequency: " << minVal << " at intensity: " << minLoc.y << endl;
    cout << "Total pixels: " << sum(hist)[0] << endl << endl;
}


Mat calculate_transformation_function(const Mat& src_img, int hist_size) {
    // Compute histogram
    Mat hist = calculate_hist(src_img, hist_size);

    // Calculate CDF
    Mat cdf = hist.clone();
    for (int i = 1; i < hist_size; i++) {
        cdf.at<float>(i) += cdf.at<float>(i-1);
    }
    cdf /= cdf.at<float>(hist_size-1);
    cdf *= 255;

    return cdf;
}


Mat histogram_equalization(const Mat& src_img, int hist_size) {
    // Compute CDF
    Mat cdf = calculate_transformation_function(src_img, hist_size);

    // Create Lookup Table
    Mat lookupTable(1, 256, CV_8U);
    uchar* p = lookupTable.ptr();
    for (int i = 0; i < 256; i++) {
        p[i] = saturate_cast<uchar>(cdf.at<float>(i));
    }

    // Apply LUT
    Mat equalized_img;
    LUT(src_img, lookupTable, equalized_img);

    return equalized_img;
}


Mat log_transform_enhance(const Mat& src_img, double c) {
    Mat float_img;
    src_img.convertTo(float_img, CV_32F);
    float_img /= 255.0;

    Mat log_img;
    log(1 + float_img, log_img);
    log_img *= c;

    log_img *= 255.0;
    Mat enhanced_img;
    log_img.convertTo(enhanced_img, CV_8U);
    return enhanced_img;
}


Mat power_law_enhance(const Mat& src_img, double c, double gamma) {
    Mat float_img;
    src_img.convertTo(float_img, CV_32F);
    float_img /= 255.0;

    Mat power_law_img;
    pow(float_img, gamma, power_law_img);
    power_law_img *= c;

    power_law_img *= 255.0;
    Mat enhanced_img;
    power_law_img.convertTo(enhanced_img, CV_8U);
    return enhanced_img;
}


int main() {

    Mat src_img = imread("../Fig308Org.tif", IMREAD_GRAYSCALE);

    if (!src_img.data) {
        cout << "Image not loaded";
        return -1;
    }

    // imwrite("../img/org_img.png", src_img);

    // Compute and display histogram of original image
    compute_and_display_histogram(src_img, "../img/org_hist.png", 256);

    Mat equalized_img = histogram_equalization(src_img, HIST_SIZE);

    compute_and_display_histogram(equalized_img, "../img/eq_hist.png", 256);
    imwrite("../img/eq_img.png", equalized_img);

    // log transform
    // Mat log_img_three = log_transform_enhance(src_img, 3);
    // imwrite("../img/log_img_3.tif", log_img_three);
    // Mat log_img_two = log_transform_enhance(src_img, 2);
    // imwrite("../img/log_img_2.tif", log_img_two);
    // Mat log_img_two_half = log_transform_enhance(src_img, 2.5);
    // imwrite("../img/log_img_25_b.png", log_img_two_half);

    // power transform
    // Mat power_img_three = power_law_enhance(src_img, 2, 0.8);
    // imwrite("../img/power_img_208_b.png", power_img_three);
    // Mat power_img_two = power_law_enhance(src_img, 2, 0.9);
    // imwrite("../img/power_img_209.tif", power_img_two);
    // Mat power_img_two_half = power_law_enhance(src_img, 2, 0.7);
    // imwrite("../img/power_img_207.tif", power_img_two_half);

    waitKey(0);

    return 0;
}