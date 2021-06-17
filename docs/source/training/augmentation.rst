.. include:: ../icons.txt

************
Augmentation
************

Augmentation is the process of creating modified versions of the training data to artificially expand the data set.
To open the Training Settings press |augmentation| **Augmentations** in the :doc:`Training Window <./nntraining>`.

.. figure:: images/augmentation.png
  :class: shadow-image
  :align: center
  
  The Augmentation Window
  
Check the **Augmentations** option to enable augmentaions. 

The **Input width** and **Input height** not necessarily relates to augmentation but defines the input size of the neural network in pixel. For some of the :doc:`applications <../applications/index>`, the input images are split into patches that are fed into the network, so a larger network input would not necessarily lead to better results.

Check **Flip horizontal** to mirror an image horizontally with 50% probability.

Check **Flip vertical** to mirror an image vertically with 50% probability.

Set the percentage to apply gaussian **Blur** to an imput image.

Set the percentage to perform **Piecewise** affine transformation to an input image.

Set the percentage to **Dropout** pixels or patches.

The **Apply Affine** percentage sets the overall probability to apply below affine transformation:

* **Scaling**: Image scaling multiplied by the factor
* **Translate**: Image translation in percentage of the image size
* **Shear**: Image shearing by degree
* **Rotation**: Rotation of the image degree

.. note::
  Image augmentation can not compensate an insufficient small data set. 
  
.. tip::
  Use augmentation with a sense of proportion. For example if your data can occur in all orientations (e.g. cells or tissue), it is perfectly fine to use flipping and rotation.

 

