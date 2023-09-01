#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <iostream>

class Sam {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::SessionOptions sessionOptions[2];
  std::unique_ptr<Ort::Session> sessionPre, sessionSam;
  Ort::RunOptions runOptionsPre;
  std::vector<int64_t> inputShapePre, outputShapePre;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<float> outputTensorValuesPre;
  std::vector<std::vector<float>> previousMasks;
  const char *inputNamesSam[6]{"image_embeddings", "point_coords", "point_labels",
                               "mask_input", "has_mask_input", "orig_im_size"},
  *outputNamesSam[3]{"masks", "iou_predictions", "low_res_masks"};

 public:
  Sam();
  ~Sam();
  void clear();
  void clearLoadModel();
  void clearPreviousMasks();
  void resizePreviousMasks(int previousMaskIdx);
  void terminatePreprocessing();
  bool loadModel(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber);
  cv::Size getInputSize();
  bool preprocessImage(const cv::Mat& image, bool *terminated);
  cv::Mat getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask);
};

#endif
