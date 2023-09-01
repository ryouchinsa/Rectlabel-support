#include "sam.h"

Sam::Sam(){}
Sam::~Sam(){
  clear();
}

void Sam::clear(){
  clearLoadModel();
  clearPreviousMasks();
}

void Sam::clearLoadModel(){
  Ort::Session* pre = sessionPre.release();
  Ort::Session* sam = sessionSam.release();
  delete pre;
  delete sam;
  inputShapePre.resize(0);
  outputShapePre.resize(0);
  outputTensorValuesPre.resize(0);
}

void Sam::clearPreviousMasks(){
  previousMasks.resize(0);
}

void Sam::resizePreviousMasks(int previousMaskIdx){
  if(previousMasks.size() > previousMaskIdx + 1){
    previousMasks.resize(previousMaskIdx + 1);
  }
}

void Sam::terminatePreprocessing(){
  runOptionsPre.SetTerminate();
}

bool modelExists(const std::string& modelPath){
  std::ifstream f(modelPath);
  if (!f.good()) {
    return false;
  }
  return true;
}

bool Sam::loadModel(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber){
  if(!modelExists(preModelPath)){
    return false;
  }
  if(!modelExists(samModelPath)){
    return false;
  }
  clearLoadModel();
  for(int i = 0; i < 2; i++){
    auto& option = sessionOptions[i];
    option.SetIntraOpNumThreads(threadsNumber);
    option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  }
  sessionPre = std::make_unique<Ort::Session>(env, preModelPath.c_str(), sessionOptions[0]);
  if(sessionPre->GetInputCount() != 1 || sessionPre->GetOutputCount() != 1){
    return false;
  }
  sessionSam = std::make_unique<Ort::Session>(env, samModelPath.c_str(), sessionOptions[1]);
  if(sessionSam->GetInputCount() != 6 || sessionSam->GetOutputCount() != 3){
    return false;
  }
  inputShapePre = sessionPre->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  outputShapePre = sessionPre->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if(inputShapePre.size() != 4 || outputShapePre.size() != 4){
    return false;
  }
  return true;
}

cv::Size Sam::getInputSize(){
  return cv::Size((int)inputShapePre[3], (int)inputShapePre[2]);
}

bool Sam::preprocessImage(const cv::Mat& image, bool *terminated){
  if(image.size() != cv::Size((int)inputShapePre[3], (int)inputShapePre[2])){
    return false;
  }
  if(image.channels() != 3){
    return false;
  }
  std::vector<uint8_t> inputTensorValues(inputShapePre[0] * inputShapePre[1] * inputShapePre[2] * inputShapePre[3]);
  for(int i = 0; i < inputShapePre[2]; i++){
    for(int j = 0; j < inputShapePre[3]; j++){
      inputTensorValues[i * inputShapePre[3] + j] = image.at<cv::Vec3b>(i, j)[2];
      inputTensorValues[inputShapePre[2] * inputShapePre[3] + i * inputShapePre[3] + j] =
          image.at<cv::Vec3b>(i, j)[1];
      inputTensorValues[2 * inputShapePre[2] * inputShapePre[3] + i * inputShapePre[3] + j] =
          image.at<cv::Vec3b>(i, j)[0];
    }
  }
  auto inputTensor = Ort::Value::CreateTensor<uint8_t>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShapePre.data(), inputShapePre.size());
  outputTensorValuesPre = std::vector<float>(outputShapePre[0] * outputShapePre[1] * outputShapePre[2] * outputShapePre[3]);
  auto outputTensorPre = Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValuesPre.data(), outputTensorValuesPre.size(), outputShapePre.data(), outputShapePre.size());
  const char *inputNamesPre[] = {"input"}, *outputNamesPre[] = {"output"};
  runOptionsPre.UnsetTerminate();
  try{
    sessionPre->Run(runOptionsPre, inputNamesPre, &inputTensor, 1, outputNamesPre, &outputTensorPre, 1);
  }catch(Ort::Exception& e){
    *terminated = true;
    std::cout<<"terminated"<<std::endl;
    return false;
  }
  return true;
}

cv::Mat Sam::getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask){
  const size_t maskInputSize = 256 * 256;
  float maskInputValues[maskInputSize];
  memset(maskInputValues, 0, sizeof(maskInputValues));
  float hasMaskValues[] = {0};
  std::vector<float> previousMaskInputValues;
  resizePreviousMasks(previousMaskIdx);
  if(isNextGetMask){
  }else if(previousMaskIdx >= 0){
    hasMaskValues[0] = 1;
    previousMaskInputValues = previousMasks[previousMaskIdx];
  }
  std::vector<float> inputPointValues, inputLabelValues;
  for(auto& point : points){
    inputPointValues.push_back((float)point.x);
    inputPointValues.push_back((float)point.y);
    inputLabelValues.push_back(1);
  }
  for(auto& point : negativePoints){
    inputPointValues.push_back((float)point.x);
    inputPointValues.push_back((float)point.y);
    inputLabelValues.push_back(0);
  }
  if(!roi.empty()){
    inputPointValues.push_back((float)roi.x);
    inputPointValues.push_back((float)roi.y);
    inputLabelValues.push_back(2);
    inputPointValues.push_back((float)roi.br().x);
    inputPointValues.push_back((float)roi.br().y);
    inputLabelValues.push_back(3);
  }
  const int numPoints = (int)inputLabelValues.size();
  std::vector<int64_t> inputPointShape = {1, numPoints, 2},
  pointLabelsShape = {1, numPoints},
  maskInputShape = {1, 1, 256, 256},
  hasMaskInputShape = {1},
  origImSizeShape = {2};
  float orig_im_size_values[] = {(float)inputShapePre[2], (float)inputShapePre[3]};
  std::vector<Ort::Value> inputTensorsSam;
  inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)outputTensorValuesPre.data(), outputTensorValuesPre.size(), outputShapePre.data(), outputShapePre.size()));
  inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
  inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputLabelValues.data(), numPoints, pointLabelsShape.data(), pointLabelsShape.size()));
  if(hasMaskValues[0] == 1){
      inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
  }else{
      inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
  }
  inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
  inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));
  Ort::RunOptions runOptionsSam;
  auto outputTensorsSam = sessionSam->Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(), inputTensorsSam.size(), outputNamesSam, 3);
  auto outputMasksValues = outputTensorsSam[0].GetTensorMutableData<float>();
  cv::Mat outputMaskSam = cv::Mat((int)inputShapePre[2], (int)inputShapePre[3], CV_8UC1);
  for (int i = 0; i < outputMaskSam.rows; i++) {
    for (int j = 0; j < outputMaskSam.cols; j++) {
      outputMaskSam.at<uchar>(i, j) = outputMasksValues[i * outputMaskSam.cols + j] > 0 ? 255 : 0;
    }
  }
  auto low_res_logits = outputTensorsSam[2].GetTensorMutableData<float>();
  previousMaskInputValues = std::vector<float>(maskInputSize);
  for (int i = 0; i < maskInputSize; i++) {
      previousMaskInputValues[i] = low_res_logits[i];
  }
  previousMasks.push_back(previousMaskInputValues);
  return outputMaskSam;
}
