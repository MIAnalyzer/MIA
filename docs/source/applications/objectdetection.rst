.. include:: <isonum.txt>

****************
Object Detection
****************

Object detection is the identification of the position of each target object in an image. Object detection can be used for quantification, counting or tracking.

.. figure:: images/objectdetection.png
  :class: shadow-image
  :align: center
  :width: 400
  
  Detection of cells (green) and dividing cells (red), original image from [#mito]_
  
Labelling
=========
To train a neural network for object detection it is necessary to create images that are labelled accordingly.

.. figure:: images/objdet_tools.png
  :class: shadow-image
  :align: center
  
  Object detection tools
  
Different tools and functions in MIA exist to label images for object detection.

.. include:: ../applications/labelling.rst

Tools
-----

By selecting the |drag| tool or pressing :kbd:`F1` you can always switch to the drag tool to zoom and change the field of view. Press the :kbd:`Spacebar` to toggle between drag tool and the last selected tool.

Use |addobject| (press :kbd:`F2`) to add objects. Press :kbd:`left_mouse` on the target object to label its position with the current active class.

With |shift| (press :kbd:`F3`) you can shift objects. Press and hold the :kbd:`left_mouse` near a labelled object and drag to move its position.

The |assign| tool (press :kbd:`F4`) can be used to asssign a new class to an existing object. Select the target class and press the :kbd:`left_mouse` close to an existing object.

The |delete| tool (press :kbd:`F5`) can be used to remove objects. Press the :kbd:`left_mouse` close to a contour to remove that contour.


Auto Detection
--------------

Press the **Auto Detection** Button to perform deep learning edge detection based on Holistically-nested edge detection [#hed]_.

.. note::
  The Auto Detection is experimental and might only work well for closed objects with a clear boundary (e.g. isolated fluorescent cells).
  
Training
========

For details about neural network training see :doc:`../training/index`. However, some settings, the network architectures, the loss functions, metrics are specific for object detection.

Object Detection Settings
-------------------------

To open the segmenation settings, press: 
|train| *Train Model* |rarr| |settings| *Detection*.

.. figure:: images/objdet_settings.png
  :class: shadow-image
  :align: center
  
  Object detection training settings
  
  
Adjust the **Object Size** slider to adapt for different object sizes that are detected. This is mostly important for distances between objects, for dense images choose a low object size and vice versa.

Change the **Object Intensity** slider to change the likelyhood of object detection. If a trained model fails to detect some objects while having only very few false positives, increase object intensity and retrain.
  
Checking **Prefer labelled parts** will **Discard up to x background tiles** in which are less than **Minimum required labelled pixels** (a pixel corresponds to an object), which can be set in the corresponding fields. For largely unbalanced datasets with a lot of background and fewer objects this option is recommended.
For unbalanced datasets see also class weighting in :doc:`../training/trainsettings`.

Neural Network architectures
----------------------------

The neural network architectures are identical to semantic segmentaion (see :ref:`segarchitectures`), but the output layer is linear that is estimated via regression.
  
Losses and Metrics
------------------

As object detection is solved as a regression problem, common regression losses and metric functions can be chosen:

Mean Squared Error
~~~~~~~~~~~~~~~~~~

.. math::
  L_{MSE} = \frac{1}{n} \sum_{i=1}^{n}{(p_i - q_i)^2}
  
with :math:`p_i` the true label and :math:`q_i` the model prediction for the :math:`i_{th}` data point.

Root Mean Squared Error
~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  L_{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n}{(p_i - q_i)^2}}.
  

Mean Squared Logarithmic Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  L_{MSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n}{(log(p_i+1) - log(q_i+1))^2}}.

Mean Absolute Error
~~~~~~~~~~~~~~~~~~~

.. math::
  L_{MAE} = \frac{1}{n} \sum_{i=1}^{n}{|p_i - q_i|}.

Postprocessing
==============

To open the postprocessing window press |postprocessing| *Postprocessing*.

.. figure:: images/objdet_pp.png
  :class: shadow-image
  :align: center
  
  Postprocessing options for object detection
  
See :doc:`../postprocessing/tracking` for a description of the tracking mode. 

Press the **Use as Stack label** button to use the currently shown contours for all frames in the currently active stack. Only applicable when using a multi-frame image stack.

  
----------

.. [#mito]
  Neumann, B., Walter, T., Hériché, J.K., Bulkescher, J., Erfle, H., Conrad, C., Rogers, P., Poser, I., Held, M., Liebel, U. and Cetin, C., 2010. Phenotypic profiling of the human genome by time-lapse microscopy reveals cell division genes. Nature, 464(7289), pp.721-727.

.. [#hed]
  Xie, S. and Tu, Z., 2015. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision (pp. 1395-1403).