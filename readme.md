# MIA - Microscopic Image Analyzer

MIA is a software for deep learning based image analysis. It covers image labeling, neural network training and inference. It can be used for image classification, object detection, semantic segmentation and tracking.

## cite

If using the software for your research, please cite:

## installation

The easiest way to install MIA is via conda (see https://docs.conda.io/en/latest/miniconda.html).

Download the [environment](https://github.com/MicroscopicImageAnalyzer/MIA/blob/master/mia_environment.yaml) file.

- Open an command line prompt
- cd to the environment file
- type: 

  - ```conda env create -f mia_environment.yml```
  - ```conda activate mia_environment```
  - ```mia_console```


There will be an installer added, soon.

## manual

The user manual can be found here.

## requirements

In general, MIA should run on any system with Linux or windows. You can use the cpu only, but it is highly recommended to have a system with a cuda-compatible gpu (from NVIDIA) to accelerate neural network training.
