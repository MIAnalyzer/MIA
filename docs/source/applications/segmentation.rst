************
Segmentation
************

Semantic image segmentation is the process of pixel-by-pixel classification that results in segments that correspond to the same object. Segmentation is typically used to identify and locate objects in an image and to determine their shape and size.


.. figure:: images/segmentation.png
  :class: shadow-image
  :align: center
  :width: 230
  
  Segmented image with synaptic spines shown in green and their parent dendrite in red
  
  
Labelling
=========
To train a neural network for semantic segmentation it is necessary to create images that are labelled accordingly.

.. figure:: images/segmentation_tools.png
  :class: shadow-image
  :align: center
  
  
  Segmentation tools
  
Different tools and functions in MIA exist to label images for semantic segmentation.

.. include:: ../applications/labelling.rst

Tools
-----

By selecting the |drag| tool you can always switch to the drag tool to zoom and change the field of view.

Use the |draw| tool to label objects with freehand labelling. While pressing the :kbd:`left_mouse` you can label target objects in the image. 

With the |poly| tool to label objects with polygones. Press the :kbd:`left_mouse` to add add polygone point. When pressing the :kbd:`right_mouse`, the first and the last set polygone points are connected resulting in the finished contour.

.. tip::
  |draw| tool and |poly| tool have different options, press the :kbd:`right_mouse` to toggle between them:
  
  * **Add** contours are added to the image from the active class. If contours overlap and are from the same class they are combined to a single contour.
  * **Delete** removes the created contour from existing contours.
  * **Slice** creates a line that is removed from existing contours, can be used to split contours.

The |assign| tool can be used to asssign a new class to an existing contour. Select the target class and press the :kbd:`left_mouse` inside an existing contour.

The |expand| tool can be used to correct existing contours. Press and hold the :kbd:`left_mouse` to correct existing contours.

.. tip::
  |expand| tool has different options, press the :kbd:`right_mouse` to toggle between them and use the :kbd:`mouse_wheel` to change the size slider:
  
  * **Expand** allows you to enlarge an existing contour or to create a new contour.
  * **Erase** removes parts from existing contours.
  * **Size** slider changes the size of the tool.
  
The |delete| tool can be used to remove contours. Press the :kbd:`left_mouse` inside a contour to remove that contour.

By checking **Inner contours**, contours may have holes. When unchecking **Internal contours** all holes inside of contours are removed, immediately.

Automated and Semi-Automated Labelling
--------------------------------------
Different automated and semi-automated segmentation option are implemented in MIA to speed-up image labelling.

.. note::
  All automated methods only affect the current field of view and ignoring everything outside. For multiple complex objects in a larger image, it can be helpful to focus the field of view on each object individually.



Smart Mode
~~~~~~~~~~

The **Smart Mode** enables an iterative, interactive segmentation mode based on grabcut [#gc]_. 
Perform smartmode segmentation as follows:

1. Select a tool (|draw|, |poly|, |expand|) to roughly label a part of the target structures
2. If parts are labelled but should not be labelled remove some of those parts that should be unlabelled
3. If parts are not labelled but should be labelled add some of those parts that should be labelled
4. Repeat steps 2 and 3
5. Select another class and repeat steps 1-4 to label objects of a different class

Auto-Segmentation
~~~~~~~~~~~~~~~~~

The **Auto Seg** function performs deep learning based edge detection based on Holistically-nested edge detection [#hed]_.

.. tip::
  * Auto-Segmentation is based on edge detection, to get proper labels you might uncheck the **Inner Contours** property.
  * Auto-Segmentation works best for isolated object with a clear boundary.
  * The auto-detected edges might also be used as a prior to further labelling 

DEXTR
~~~~~

**DEXTR** is deep learning based method that uses the objects extreme points to label the target object [#dextr]_.

Press the :kbd:`left_mouse` on the 4 extreme points (top, left, right, bottom) of the target object and that object is segmented and assigned the active class. 
The order of the points is irrelevant.


Training
========


----------

.. [#gc]
  Rother, C., Kolmogorov, V. and Blake, A., 2004. " GrabCut" interactive foreground extraction using iterated graph cuts. ACM transactions on graphics (TOG), 23(3), pp.309-314.
  
.. [#hed]
  Xie, S. and Tu, Z., 2015. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision (pp. 1395-1403).
  
.. [#dextr]
  Maninis, K.K., Caelles, S., Pont-Tuset, J. and Van Gool, L., 2018. Deep extreme cut: From extreme points to object segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 616-625).