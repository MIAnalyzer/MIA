from setuptools import setup, find_packages
import os


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

here = os.path.abspath(os.path.dirname(__file__))
ver = {} 
with open(os.path.join(here, 'mia', '__version__.py')) as f:
    exec(f.read(), ver)

setup(
    name="mianalyzer", 
    version=ver['__version__'],
    author="Nils Koerber",
    author_email="nils.koerber@t-online.de",
    description="MIA deep learning based Microscopic Image Analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",

	install_requires=[
	"scipy",
	"scikit_image",
	"numpy",
	"imgaug",
	"opencv_python",
	"tensorflow",
	"matplotlib",
	"Pillow",
	"PyQt5",
	"keras-applications"
	],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
	packages=find_packages(include=[
        "mia",
        "mia.*"
    ]),
	entry_points={
        'console_scripts': [
            'mia_console = mia.startup:start',
        ],
		'gui_scripts': [
            'mia = mia.startup:start',
        ],
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
	]
	
	),
     ],

	include_package_data=True,
	
    python_requires=">=3.6",
)