segmentation 

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

By checking **Internal contours**, contours may have holes. When unchecking **Internal contours** all holes inside of contours are removed, immediately.

Smartmode
---------

The **Smartmode** enables an iterative, interactive segmentation mode based on grabcut [#gc]_. 
Perform smartmode segmentation as follow:

1. Select a tool (|draw|, |poly|, |expand|) to label a part of the target structures
2. Remove parts that where labelled and should not be labelled
3. Repeat steps 1 and 2
4. Select another class and repeat steps 1-3 to label objects of a different class






.. [#gc]
  Rother, Carsten, Vladimir Kolmogorov, and Andrew Blake. "" GrabCut" interactive foreground extraction using iterated graph cuts." ACM transactions on graphics (TOG) 23.3 (2004): 309-314.