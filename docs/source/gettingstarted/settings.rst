.. include:: <isonum.txt>

********
Settings
********

In the Settings window all global settings can be made.

.. figure:: images/settings.png
  :class: shadow-image
  :align: center
  
  The Settings Window
  
Deep Learning Settings
======================
 
In the **GPU settings**, the engine for deep learning training and prediction can be set. Select the graphics processing unit (GPU), multiple GPUs or CPU that you want to use for deep learning. 

.. tip::
	Due to the high degree of parallelization deep learning training is strongly accelerated by using a GPU. See :doc:`../introduction/installation` to check if you have a compatible GPU.
	
**Worker threads** sets the number of parallel threads used for computation. By default it is set to the half of the maximum of CPU threads that are identified on the running hardware system.

**Image Down Scaling** can be set to decrease the image input size for all computations performed but not for visualization. If you set the image down scaling below 1, the image will be scaled to the width and height multiplied by the set factor before any operations. By reducing down scaling the processing speed can be increased but the accuracy might be worse compared to full size image input.

Display Settings
================

Check **Show Object Numbers** to display the object number of each displayed shape.

**Fast drawing** enhances drawing speed of objects on the image. Usually it is not necessary to check the Fast drawing option, however if you have huge amounts of object or notice software delay during image labelling, Fast drawing might improve experience. 

**Font Size** sets the font size of all text, such as shape numbers, written to the image.

**Pen Size** sets the line thickness of all objects drawn to the image. The best Pen Size might depend on image resolution as all drawings are depending on input image resolution.

**Contour Opacity** sets the opacity of each contour, with the slider in the leftmost position rendering shapes opaque and the slider in the rightmost position rendering shapes transparent.

.. note::
	All settings in the display settings take effect only after the window is closed.
	
Load/Save Settings
==================

All settings (not only from the settings window) are saved upon closing the program and loaded at startup. To save settings explicitly, go to *File* |rarr| *Save Settings* and use *File* |rarr| *Load Settings* to load settings.