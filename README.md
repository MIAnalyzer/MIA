# MIA - Microscopic Image Analyzer
![MIA](https://github.com/MIAnalyzer/MIA/blob/master/docs/source/gettingstarted/images/user_interface.PNG?raw=true)

MIA is a software for deep learning based image analysis. It covers image labeling, neural network training and inference. It can be used for image classification, object detection, semantic segmentation and tracking.

## Installation
The easiest way to install MIA is via conda (see https://docs.conda.io/en/latest/miniconda.html for installation options).

After installation of conda, download the [environment](https://github.com/MIAnalyzer/MIA/releases/download/v0.2.4/mia_environment.yaml) file. 

Then, open an anaconda prompt and type:
- ```cd /path/to/mia_environment.yaml```  (change ```/path/to/``` to the path of the folder with the environment file)
- ```conda env create -f mia_environment.yaml```
- wait and follow instructions
  
### to start the software 
type in an anaconda prompt:
  - ```conda activate mia_environment```
  - ```mianalyzer```

## How to get help?

A quickstart guide can be found [here](https://mianalyzer.github.io/gettingstarted/quickstart.html) and the complete user manual [here](https://mianalyzer.github.io/).

Please use [image.sc](https://forum.image.sc/tag/mia) with the ```mia```-tag for general discussion, questions about how to use the software or feature requests. Bugs can be reported directly in the [issues](https://github.com/MIAnalyzer/MIA/issues) panel on github.

## Reference
If you use this code for your research, please cite: 

https://biorxiv.org/cgi/content/short/2022.01.14.476308v1


## Requirements

In general, MIA should run on any system with Linux or windows. You can use the cpu only, but it is highly recommended to have a system with a [cuda-compatible](https://developer.nvidia.com/cuda-gpus) gpu (from NVIDIA) to accelerate neural network training.

