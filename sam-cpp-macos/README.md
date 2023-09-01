## Segment Anything CPP Wrapper

This project is to create a pure C++ inference api for [Segment Anything](https://github.com/facebookresearch/segment-anything) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), with no dependence on Python during runtime. The code repository contains a C++ library with a test program to facilitate easy integration of the interface into other projects.

Model loading takes approximately 10 or 1 seconds, and a single inference takes around 20ms, obtained using Intel Xeon W-2145 CPU (16 threads). During runtime, the interface may consume around 6GB or 1GB memory if running on CPU, and 16GB or 1GB if running on CUDA. The "or" here means values for "Segment Anything" or "MobileSAM".

Currently, this program has been thoroughly tested on Windows and may encounter issues when running on Linux (no pre-compiled Linux version is provided).

### Test program - sam_cpp_test
![](demo.jpg)

Video demo [link](https://youtu.be/6NyobtZoPKc)

Image by <a href="https://pixabay.com/users/brenda2102-30343687/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=7918031">Brenda</a> from <a href="https://pixabay.com//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=7918031">Pixabay</a> and key recorded by [KeyCastOW](https://github.com/brookhong/KeyCastOW)

Usage:

Download compressed file in the release page, unzip it, and run sam_cpp_test directly or in command line:

```bash
# Show help
./sam_cpp_test -h

# Example (change device, use CPU for preprocess and CUDA for sam)
# If you have multiple GPUs, you can use CUDA:1, CUDA:2, etc.
# All in cpu or all in cuda is also supported
./sam_cpp_test -pre_device="cpu" -sam_device="cuda:0"

# Example (default options)
./sam_cpp_test -pre_model="models/sam_preprocess.onnx" -sam_model="models/sam_vit_h_4b8939.onnx" -image="images/input.jpg"

# Example (use MobileSAM)
./sam_cpp_test -pre_model="models/mobile_sam_preprocess.onnx" -sam_model="models/mobile_sam.onnx"

# Example (change image)
./sam_cpp_test -image="images/input2.jpg"
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
