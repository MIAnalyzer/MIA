************
Installation
************


The easiest way to install MIA is via conda (see https://docs.conda.io/en/latest/miniconda.html for installation options).

After installation of conda, download the `environment <https://github.com/MIAnalyzer/MIA/releases/download/weights/mia_environment.yaml>`_ file.

Then, open an anaconda prompt and type:

- Open an anaconda prompt
- ``cd /path/to/mia_environment.yaml`` (change '/path/to/' to the path of the folder with the environment file)
- ``conda env create -f mia_environment.yaml``
  
**to start the software:**

type in an anaconda prompt:

- ``conda activate mia_environment``
- ``mianalyzer``