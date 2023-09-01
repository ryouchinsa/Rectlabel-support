## Segment Anything CPP Wrapper for macOS

This code is originated from [Segment Anything CPP Wrapper](https://github.com/dinglufe/segment-anything-cpp-wrapper) and customized to use low_res_logits which is the previous mask result and to use CPU mode for macOS.

Download a zipped model folder from
[MobileSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/mobile_sam.zip), [ViT-Large SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195.zip), and [ViT-Huge SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_h_4b8939.zip).
Put the unzipped model folder into sam-cpp-macos folder.

Edit the modelName in [test.cpp](https://github.com/ryouchinsa/Rectlabel-support/blob/master/sam-cpp-macos/test.cpp).
```cpp
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
```

After loading the model, the preprocessing for the image begins. Because of CPU mode, it takes 2 seconds for "MobileSAM", 30 seconds for "ViT-Large SAM", and 60 seconds for "ViT-Huge SAM" on the Apple M1 device.
```cpp
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
```
Build and run.
```bash
cmake -S . -B build
cmake --build build
./build/sam_cpp_test
```
