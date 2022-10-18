# lern-deep-lerning
lerning deep lerning

# Python CUDA set up on Windows 10 for GPU support

## Software Used:

````
- VS2017 (Visual Studio Community 2017)
- cuda_10.0.130_411.31_win10.exe
- cudnn_10.0-windows-x64-v7.4.2.24
````

## Step 1: Check Configuration and Compatibility

````
https://www.tensorflow.org/install/source_windows
````

It shows the compatibility between Tensorflow, Python, CUDA and CUDNN version


## Step 2: Install VS2017

http://www.visualstudio.com/vs/older-downloads

````
vs_enterprise.exe --layout "C:\myFolder\vs2017" --lang en-US
````

## Step 3: Install CUDA Toolkit

https://developer.nvidia.com/cuda-10.0-download-archive

````
Choose the following target platform:
- Operating System: Windows
- Architecture: x86_64
- Version: 10
- Installation Type: exe (local)
````

Copy and install it in your remote computer. Note that after installation, environment variables (i.e. CUDA_PATH and CUDA_PATH_V10_0) will be created automatically


## Step 4: Install cuDNN

````
https://developer.nvidia.com/rdp/cudnn-archive

Choose `Download cuDNN v7.4.2 (Dec 14, 2018) for CUDA 10.0` followed by `cuDNN Library for Windows 10`.

Unzip and copy the folder to your remote computer.


Go to your Environment Variables and search the path of CUDA_PATH from System variables. In my case, it is in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0. Go to this CUDA installation folder.

Copy the content inside bin, include and lib of the cuDNN folder to the CUDA installation folder (i.e. bin\cudnn64_7.dll to the bin\, include\cudnn.h to the include\ folder and lib\x64\cudnn.lib to the lib\x64\ folder).


From step 5 onwards, I will recommend doing them in virtual conda environment, so that in cases where there are multiple project environments, modifying Python libraries in a conda environment will not affect other projects. The conda environment can also be easily portable (copy-and-paste) to remote computers.
````

## Step 5: Install Tensorflow GPU
````
pip install tensorflow-gpu==1.14.0
````

## Step 6: Install Keras
````
pip install Keras
````
## Step 7: Install Pytorch (Optional)

````
pip install torch==1.3.1 torchvision==0.4.2 -f https://download.pytorch.org/whl/torch_stable.html


Alternatively, you can find the command from Pytorch website: https://pytorch.org/get-started/locally. Choose the following (as at time of writing):
- Pytorch Build: Stable (1.3)
- Your OS: Windows
- Package: PIP
- Language: Python 3.6
- CUDA: 10.1 (v10.0 no longer available at time of writing)
````

## Step 8: Verify installation

````
# Step 1: Check Pytorch (optional)
import torch
print("Cuda available: ", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name())
# Step 2: Check Tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# Step 3: Check Keras (optional)
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
````
