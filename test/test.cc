#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

cv::Vec3b adjust(cv::Vec3b pixal, float adapted){
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;
    cv::Vec3b color = pixal;
    //printf("%d %d %d \n",pix[0],pix[1],pix[2]);
    for(int c = 0; c < 3; c++){
        float tmp = color[c];
        tmp *= adapted;
        tmp = (tmp * (A * tmp + B)) / (tmp * (C * tmp + D) +E) * tmp;
        tmp = tmp < 0 ? 0:tmp;
        tmp = tmp > 255 ? 255:tmp;
        color[c] = tmp;
    }
    //printf("%d %d %d \n",pix[0],pix[1],pix[2]);
    return color;
}

cv::Mat adjust_image(cv::Mat img, float adapted){
    cv::Mat img_proc = img.clone();
    int width = img.cols;
    int height = img.rows;
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            img_proc.ptr<cv::Vec3b>(i)[j] = adjust(img.ptr<cv::Vec3b>(i)[j], adapted);
        }
    }
    return img_proc;
}

/*计算两张图像的PSNR*/
void compute_psnr(const cv::Mat img1, const cv::Mat img2, double& psnr){
    cv::Mat diff;
    absdiff(img1, img2, diff);  // |I1 - I2|
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);  // |I1 - I2|^2
    cv::Scalar s = sum(diff);

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10){
        psnr = 0;
    }
    else{
        double  mse = sse /(double)(img1.channels() * img1.total());
        psnr = 10.0*log10((255*255)/mse);
    }
}

/*图像信息熵也是图像质量评价的常用指标, 通常情况下，
 * 图像信息熵越大,其信息量就越丰富，质量越好。*/
void compute_entropy(cv::Mat img, double& entropy){
    double temp[256] = { 0.0f };
    // 计算每个像素的累积值
    int row = img.rows;
    int col = img.cols;
    for (int r = 0; r<row; r++){
        for (int c = 0; c<col; c++){
            const uchar * i = img.ptr<uchar>(r,c);
            temp[*i] ++;
        }
    }

    // 计算每个像素的概率
    int size = row * col;
    for (int i = 0; i<256; i++){
        temp[i] = temp[i] / size;
    }

    double result = 0.0f;
    // 计算图像信息熵
    for (int i = 0; i<256; i++){
        if (temp[i] != 0.0) {
            result += temp[i] * log2(temp[i]);
        }
    }
    entropy = -result;
}

/*计算灰度图的均值和方差*/
void compute_mean_std(const cv::Mat img, double & mean, double & std) {
    cv::Mat gray;
    if (img.channels() != 1) {
        cv::cvtColor(img, gray, CV_BGR2GRAY);
    }
    else{
        gray = img;
    }
    cv::Mat mat_mean, mat_stddev;
    meanStdDev(gray, mat_mean, mat_stddev);
    mean = mat_mean.at<double>(0, 0);
    std = mat_stddev.at<double>(0, 0);
}

int main(void){

    std::string img_path = "../test_images/";
    std::string img_list = img_path + "imglist.txt";
    std::ifstream ifs(img_list);
    std::string result_path = "../output_images/";
    if (ifs.fail()){
            std::cout << "image list " << img_list  << " not find "<< std::endl;
    }
    std::string img_name;
    while(ifs >> img_name){
        std::cout << img_name << std::endl;
        std::string img_full_name = img_path + img_name;

        cv::Mat img = cv::imread(img_full_name);
        cv::Mat img_ref = img.clone();
        double entropy_ref = 0, mean_ref = 0, std_ref = 0;
        float adapted_opt = 0;
        for(float adapted = 0.1; adapted < 1.3; adapted += 0.1){
            cv::Mat img_proc = adjust_image(img, adapted);
            cv::imwrite("./out.jpg", img_proc);
            system("../imgcat.sh out.jpg");
            double entropy, mean, std;
            // compute_psnr(img_ref, img_proc, psnr);
            compute_entropy(img_proc, entropy);
            compute_mean_std(img_proc, mean, std);
            //printf("adapted =%f entroy = %f mean =%f std = %f \n",adapted, entropy, mean, std);
            if( entropy > entropy_ref){ //交叉熵评判图像质量
                img_ref = img_proc.clone();
                entropy_ref = entropy;
                adapted_opt = adapted;
            }
        }
        printf("### process %s finish best adapted = %f \n", img_name.c_str(), adapted_opt);
        cv::imwrite(result_path + img_name, img_ref);
        //system("~imgcat.sh " + result_path + img_name);
    }
}
