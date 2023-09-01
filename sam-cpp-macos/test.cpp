#include <opencv2/opencv.hpp>
#include <thread>
#include "sam.h"

int main(int argc, char** argv) {
  Sam sam;
  std::string modelName = "mobile_sam";
  std::string pathEncoder = modelName + "/" + modelName + "_preprocess.onnx";
  std::string pathDecoder = modelName + "/" + modelName + ".onnx";
  std::cout<<"loadModel started"<<std::endl;
  bool successLoadModel = sam.loadModel(pathEncoder, pathDecoder, std::thread::hardware_concurrency());
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  std::string imagePath = "david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg";
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  auto inputSize = sam.getInputSize();
  cv::resize(image, image, inputSize);
  cv::imwrite("resized.jpg", image);
  bool terminated = false; // Check the preprocessing is terminated when the image is changed
  std::cout<<"preprocessImage started"<<std::endl;
  bool successPreprocessImage = sam.preprocessImage(image, &terminated);
  if(!successPreprocessImage){
    std::cout<<"preprocessImage error"<<std::endl;
    return 1;
  }
  std::cout<<"getMask started"<<std::endl;
  std::list<cv::Point> points, nagativePoints;
  cv::Rect roi;
  // 1st object and 1st click
  int previousMaskIdx = -1; // An index to use the previous mask result
  bool isNextGetMask = true; // Set true when start labeling a new object
  points.push_back({810, 550});
  cv::Mat mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object1-click1.png", mask);
  // 1st object and 2nd click
  isNextGetMask = false;
  points.push_back({940, 410});
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object1-click2.png", mask);
  // 1st object and 3rd click
  points.push_back({855, 405});
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object1-click3.png", mask);
  // 1st object and 4th click
  points.push_back({755, 415});
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object1-click4.png", mask);
  // 1st object and 5th click
  points.push_back({710, 470});
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object1-click5.png", mask);
  // 2nd object and 1st click
  isNextGetMask = true;
  points.clear();
  points.push_back({815, 200});
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object2-click1.png", mask);
  // 2nd object and 2nd click
  isNextGetMask = false;
  points.push_back({755, 310});
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object2-click2.png", mask);
  // 2nd object and 3rd click
  points.push_back({715, 250});
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object2-click3.png", mask);
  // 1st object and box
  isNextGetMask = true;
  points.clear();
  roi = cv::Rect(681, 386, 320, 260);
  mask = sam.getMask(points, nagativePoints, roi, previousMaskIdx, isNextGetMask);
  previousMaskIdx++;
  cv::imwrite("mask-object1-box.png", mask);
  return 0;
}
