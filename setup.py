from setuptools import setup, find_packages
import os



here = os.path.abspath(os.path.dirname(__file__))
ver = {} 
with open(os.path.join(here, 'mia', '__version__.py')) as f:
    exec(f.read(), ver)

setup(
    name="mianalyzer", 
    version=ver['__version__'],
    author="Nils Koerber",
    description="MIA deep learning based Microscopic Image Analyzer",
	long_description="MIA is a software for deep learning based image analysis. It covers image labeling, neural network training and inference. It can be used for image classification, object detection, semantic segmentation and tracking, see https://github.com/MIAnalyzer/MIA/ for details.",
    url="https://github.com/MIAnalyzer/MIA/",

	install_requires=[
	# requirements removed here as MIA installation via conda is recommended
    # when installing via pip use requirements.txt
	],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
	packages=find_packages(include=[
        "mia",
        "mia.*"
    ]),
	entry_points={
        'console_scripts': [
            'mianalyzer = mia.startup:start',
        ],
        # downloads do not work without console -> fix
		#'gui_scripts': [
        #    'mia = mia.startup:start',
        #],
	},
	data_files=[('icons', [
	'mia/icons/addobject.png', 
	'mia/icons/assign.png',
	'mia/icons/assignclass.png', 
	'mia/icons/augmentation.png',
	'mia/icons/clear.png', 
	'mia/icons/delete.png',
	'mia/icons/dextr.png', 
	'mia/icons/drag.png',
	'mia/icons/draw.png', 
	'mia/icons/expand.png',
	'mia/icons/exportallmasks.png', 
	'mia/icons/exportmask.png',
	'mia/icons/load.png', 
	'mia/icons/loadmodel.png',
	'mia/icons/magicwand.png',
	'mia/icons/measure.png', 
	'mia/icons/next.png',
	'mia/icons/objectcolor.png',
	'mia/icons/objectnumber.png', 
	'mia/icons/poly.png',
	'mia/icons/postprocessing.png', 
	'mia/icons/predict.png',
	'mia/icons/predictall.png', 
	'mia/icons/previous.png',
	'mia/icons/reset.png',
	'mia/icons/results.png', 
	'mia/icons/saveall.png',
	'mia/icons/savemodel.png', 
	'mia/icons/setclass.png',
	'mia/icons/settings.png', 
	'mia/icons/shift.png',
	'mia/icons/tracking.png', 
	'mia/icons/train.png',
	'mia/icons/logo.png', 
	'mia/icons/loading.png', 
	'mia/dl/machine_learning/deploy.prototxt',
	]
	
	),
     ],

	include_package_data=True,
	
    python_requires=">=3.6",
)