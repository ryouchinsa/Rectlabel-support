## Segment Anything CPP Wrapper for macOS

This code is originated from [Segment Anything CPP Wrapper](https://github.com/dinglufe/segment-anything-cpp-wrapper) and customized to use low_res_logits which is the previous mask result and to use CPU mode for macOS.

Download a zipped model file from
[MobileSAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/mobile_sam.zip), [ViT-Large SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_l_0b3195.zip), and [ViT-Huge SAM](https://huggingface.co/rectlabel/segment-anything-onnx-models/resolve/main/sam_vit_h_4b8939.zip).
Put the unzipped model folder into sam-cpp-macos folder.

```bash
cmake -S . -B build
cmake --build build
./build/sam_cpp_test
```

### C++ library - sam_cpp_lib

A simple example:

```cpp
Sam::Parameter param("sam_preprocess.onnx", "sam_vit_h_4b8939.onnx", std::thread::hardware_concurrency());
param.providers[0].deviceType = 0; // cpu for preprocess
param.providers[1].deviceType = 1; // CUDA for sam
Sam sam(param);

// Use MobileSAM
Sam::Parameter param("mobile_sam_preprocess.onnx", "mobile_sam.onnx", std::thread::hardware_concurrency());

auto inputSize = sam.getInputSize();
cv::Mat image = cv::imread("input.jpg", -1);
cv::resize(image, image, inputSize);
sam.loadImage(image); // Will require 6GB memory if using CPU, 16GB if using CUDA

// Using SAM with prompts (input: x, y)
cv::Mat mask = sam.getMask({200, 300});
cv::imwrite("output.png", mask);

// Using SAM with multiple prompts (input: points, nagativePoints)
cv::Mat mask = sam.getMask(points, nagativePoints); //Will require 1GB memory/graphics memory
cv::imwrite("output-multi.png", mask);

// Using SAM with box prompts (input: points, nagativePoints, box)
// The points and negativePoints can be empty (use {} as parameter)
cv::Rect box{444, 296, 171, 397};
cv::Mat mask = sam.getMask(points, nagativePoints, box);
cv::imwrite("output-box.png", mask);

// Automatically generating masks (input: number of points each side)
// Slow since running on CPU and the result is not as good as official demo
cv::Mat maskAuto = sam.autoSegment({10, 10});
cv::imwrite("output-auto.png", maskAuto);
```

More details can be found in [test.cpp](test.cpp) and [sam.h](sam.h).

The "sam_vit_h_4b8939.onnx" and "mobile_sam.onnx" model can be exported using the official steps in [here](https://github.com/facebookresearch/segment-anything#onnx-export) and [here](https://github.com/ChaoningZhang/MobileSAM#onnx-export). The "sam_preprocess.onnx" and "mobile_sam_preprocess.onnx" models need to be exported using the [export_pre_model](export_pre_model.py) script (see below).

### Export preprocessing model

Segment Anything involves several [preprocessing steps](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb), like this:

```Python
sam.to(device='cuda')
predictor = SamPredictor(sam)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
```

The [export_pre_model](export_pre_model.py) script exports these operations as an ONNX model to enable execution independent of the Python environment. One limitation of this approach is that the exported model is dependent on a specific image size, so subsequent usage will require scaling images to that size. If you wish to modify the input image size (longest side not exceed 1024), the preprocessing model must be re-exported. Running the script requires installation of the [Segment Anything](https://github.com/facebookresearch/segment-anything#getting-started) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM#getting-started), and it requires approximately 23GB or 2GB of memory during execution for "Segment Anything" or "MobileSAM" respectively.

The [export_pre_model](export_pre_model.py) script needs to be modified to switch between Segment-anything and MobileSAM:

```Python
# Uncomment the following lines to generate preprocessing model of Segment-anything
# import segment_anything as SAM
# # Download Segment-anything model "sam_vit_h_4b8939.pth" from https://github.com/facebookresearch/segment-anything#model-checkpoints
# # and change the path below
# checkpoint = 'sam_vit_h_4b8939.pth'
# model_type = 'vit_h'
# output_path = 'models/sam_preprocess.onnx'
# quantize = True

# Uncomment the following lines to generate preprocessing model of Mobile-SAM
# Download Mobile-SAM model "mobile_sam.pt" from https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt
checkpoint = 'mobile_sam.pt'
model_type = 'vit_t'
output_path = 'models/mobile_sam_preprocess.onnx'
quantize = False
```

```bash

```bash

### Build

First, install the dependencies in [vcpkg](https://vcpkg.io):

#### Windows

```bash
./vcpkg install opencv:x64-windows gflags:x64-windows onnxruntime-gpu:x64-windows
```

Then, build the project with cmake.
```bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
```

#### Linux

Download [onnxruntime-linux-x64-1.14.1.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz)

```bash
./vcpkg install opencv:x64-linux gflags:x64-linux
```

build the project with cmake.

```bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake -DONNXRUNTIME_ROOT_DIR=[onnxruntime-linux-x64-1.14.1 root]
```

### License

MIT
