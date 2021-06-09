.. include:: <isonum.txt>

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

By selecting the |drag| tool or pressing :kbd:`F1` you can always switch to the drag tool to zoom and change the field of view. Press the :kbd:`Spacebar` to toggle between drag tool and the last selected tool.

Use the |draw| tool (press :kbd:`F2`) to label objects with freehand labelling. While pressing the :kbd:`left_mouse` you can label target objects in the image. 

With the |poly| tool (press :kbd:`F3`) to label objects with polygones. Press the :kbd:`left_mouse` to add add polygone point. When pressing the :kbd:`right_mouse`, the first and the last set polygone points are connected resulting in the finished contour.

.. tip::
  |draw| tool and |poly| tool have different options, press the :kbd:`right_mouse` to toggle between them:
  
  * **Add** contours are added to the image from the active class. If contours overlap and are from the same class they are combined to a single contour.
  * **Delete** removes the created contour from existing contours.
  * **Slice** creates a line that is removed from existing contours, can be used to split contours.

The |assign| tool (press :kbd:`F4`) can be used to asssign a new class to an existing contour. Select the target class and press the :kbd:`left_mouse` inside an existing contour.

The |expand| tool (press :kbd:`F5`) can be used to correct existing contours. Press and hold the :kbd:`left_mouse` to correct existing contours.

.. tip::
  |expand| tool has different options, press the :kbd:`right_mouse` to toggle between them and use the :kbd:`mouse_wheel` to change the size slider:
  
  * **Expand** allows you to enlarge an existing contour or to create a new contour.
  * **Erase** removes parts from existing contours.
  * **Size** slider changes the size of the tool.
  
The |delete| tool (press :kbd:`F6`) can be used to remove contours. Press the :kbd:`left_mouse` inside a contour to remove that contour.

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

For details about neural network training see :doc:`../training/index`. However, some settings, the network architectures, the loss functions, metrics are specific for segmentation.

Segmentation Settings
---------------------

To open the segmenation settings, press: 
|train| *Train Model* |rarr| |settings| *Segmentation*.

.. figure:: images/segmentation_settings.png
  :class: shadow-image
  :align: center
  
  Segmentation training settings
  
Checking **Separate Contours** will weight pixels in the proximity of 2 contours higher, increasing the likelyhood that contours that are close to each other will be separated [#unet]_. Recommended for a dataset with an high density of objects of the same class or with clustered objects.
This option might decrease training speed as the pixels weights need to be calculated for each image.

Checking **Prefer labelled parts** will **Discard up to x background tiles** in which are less than **Minimum required labelled pixels**, which can be set in the corresponding fields. For largely unbalanced datasets with a lot of background and fewer objects this option is recommended.
For unbalanced datasets see also class weighting in :doc:`../training/trainsettings`.

.. _segarchitectures:

Neural Network architectures
----------------------------

Most neural networks for semantic segmentation are based on a Fully Convolutional Network architecture ommiting fully connected layers [#fcn]_. 
During the training process patches of the input images are generated and fed into the network and compared to the corresponding label patch. The size of those patches can be set in :doc:`../training/augmentation`.

The following Network architectures are currently supported, please refer to the papers for details:

* U-Net [#unet]_
* Feature Pyramid Network (FPN) [#fpn]_
* LinkNet [#lnkn]_
* Pyramid Scene Parsing Network (PSPNet) [#psp]_
* DeepLabv3+ [#dlv3]_

The following backbones are currently supported, untrained or with pretrained weights pre-trained on imagenet dataset [#imn]_:


=====================  =======================================================================================================================================  ========
**Model**              **Options**                                                                                                                              **Ref.**
---------------------  ---------------------------------------------------------------------------------------------------------------------------------------  --------
DenseNet               densenet121, densenet169, densenet201                                                                                                    [#dns]_
EfficientNet           efficientnetb0, efficientnetb1, efficientnetb2, efficientnetb3, efficientnetb4, efficientnetb5, efficientnetb6, efficientnetb7           [#eff]_
Inception              inceptionv3, inceptionresnetv2                                                                                                           [#inc]_
MobileNet              mobilenet, mobilenetv2                                                                                                                   [#mob]_
ResNet                 resnet18, resnet34, resnet50, resnet101, resnet152                                                                                       [#res]_
ResNeXt                resnext50, resnext101                                                                                                                    [#rsx]_
SE-ResNet              seresnet18, seresnet34, seresnet50, seresnet101, seresnet152                                                                             [#sen]_
SE-ResNeXt             seresnext50, seresnext101                                                                                                                [#sen]_
SENet                  senet154                                                                                                                                 [#sen]_
VGG	                   vgg16, vgg19                                                                                                                             [#vgg]_
Xception               xception                                                                                                                                 [#xcp]_
=====================  =======================================================================================================================================  ========

.. note::
  Not all model backbones are available for all model architectures.
 
.. tip::
  * The U-Net is a very popular choice for segmentation, it might be a good starting point as network architecture.
  * Generally the numbers behind the backbone architecture gives either the number of convolutional layers (e.g. resnet18) or the model version (e.g. inceptionv3). 
  * When you have limited computing recources use a small network architecture or a network optimized for efficiency (e.g. mobilenetv2).
  * From the supported network-backbones the senet154 shows the highest performance on imagenet classification and the slowest processing time.
  * From the supported network-backbones the mobilenet has the fastest processing time and fewest parameters.


Losses and Metrics
------------------

For semantic segmentation several objective function have been tested for neural network optimization and directly impact the model training.
Metrics are used to measure the performance of the trained model, but are independent of the optimization and the training process.
The loss and metric functions can be set in  |train| *Train Model* |rarr| |settings| *Settings*.

Cross Entropy
~~~~~~~~~~~~~

The cross entropy loss is a widely used objective function used for classification and segmentation (which is per pixel classification). It is defined as:

.. math::
  L_{CE} = -\sum_{i=1}^{n}{p_i log(q_i)},
  
with :math:`p_i` the true label and :math:`q_i` the model prediction for the :math:`i_{th}` class.

Focal Loss
~~~~~~~~~~

The focal loss is an extension of the cross entropy, which improves performance for unbalanced datasets [#fl]_. It is defined as follows:

.. math::
  L_{FL} = -\sum_{i=1}^{n}{(1-q_i)^\gamma p_i log(q_i)},

with :math:`\gamma` as the focussing parameter. Default is set :math:`\gamma = 2`.
  
Kullback-Leibler Divergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Kullback-Leibler Divergence, sometimes referred as relative entropy, is defined as follows:

.. math::
  L_{KL} = -\sum_{i=1}^{n}{p_i (log(p_i)-log(q_i))}.
  
  
Dice Loss
~~~~~~~~~

The dice loss, which can be used for segmentation of highly imbalanced data [#vnet]_ and is defined as follows:

.. math::
  L_{Dice} = -\frac{2 \sum_{i=1}^{n}{p_i q_i}}{\sum_{i=1}^{n}{p_i} + \sum_{i=1}^{n}{q_i}}.

Intersection over Union
~~~~~~~~~~~~~~~~~~~~~~~

The intersection over union is very similar to the dice coefficient and a measure for the overlap of the prediction and the ground truth. It is a widely used measure for segmentation model performance and is defined as follows:

.. math::
  L_{iou} = \frac{\sum_{i=1}^{n}{p_i q_i}}{\sum_{i=1}^{n}{p_i} + \sum_{i=1}^{n}{q_i} - \sum_{i=1}^{n}{p_i q_i}}.
  
Pixel Accuracy
~~~~~~~~~~~~~~

The pixel accuracy measures all pixels that are classified correctly:

.. math::
  L_{acc} = \frac{t_p + t_n}{t_p + t_n + f_p + f_n},
  
with :math:`t_p` the true positives (:math:`p_i=1` and :math:`q_i=1`), :math:`t_n` the true negatives (:math:`p_i=0` and :math:`q_i=0`), :math:`f_p` the false positives (:math:`p_i=0` and :math:`q_i=1`) and :math:`f_n` the false negatives (:math:`p_i=1` and :math:`q_i=0`). 
The pixel accuracy is a misleading measure for imbalanced data.


.. tip::
  A good starting point for choosing a **loss function** is usually the cross entropy loss. When you have imbalanced data, you can switch to focal loss. Dice loss should be used for special cases only, as gradients (and with that the general training) are more unstable.
  As **metric function** the intersection over union is for most cases a valid choice.


Postprocessing
==============

To open the postprocessing window press |postprocessing| *Postprocessing*.

.. figure:: images/segmentation_pp.png
  :class: shadow-image
  :align: center
  
  Postprocessing options for segmentation
  
See :doc:`../postprocessing/tracking` for a description of the tracking mode. 

The **min Contour Size** option can be used to dismiss all contours that have a smaller size than the given pixels.

By checking **Show Skeleton** the skeleton of each contour is calculated. The slider can be used to set contour smoothing before skeleton calculation. 

.. note::
  The skeleton is dynamically calculated. To avoid unecessary waiting, turn off skeleton calculation when working with many or complex contours or making changes to contours.

Press the **Use as Stack label** button to use the currently shown contours for all frames in the currently active stack. Only applicable when using a multi-frame image stack.


Results
=======

See :doc:`../results/index` for more information.

To open the result settings for segmentation press |settings| *Settings* in the results window.

.. figure:: images/segmentation_ressettings.png
  :class: shadow-image
  :align: center
	
  Export options for segmentation
  
You can select the export options for segmentation. You have the option to export **Skeleton** size, the **Perimeter** for each contour, the **Min Intensity** minimum intensity value inside each contour, the **Mean Intensity** mean intensity value inside each contour and the **Max Intensity** maximum intensity value inside each contour. 


----------

.. [#gc]
  Rother, C., Kolmogorov, V. and Blake, A., 2004. " GrabCut" interactive foreground extraction using iterated graph cuts. ACM transactions on graphics (TOG), 23(3), pp.309-314.
  
.. [#hed]
  Xie, S. and Tu, Z., 2015. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision (pp. 1395-1403).
  
.. [#dextr]
  Maninis, K.K., Caelles, S., Pont-Tuset, J. and Van Gool, L., 2018. Deep extreme cut: From extreme points to object segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 616-625).
  
.. [#unet]
  Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.
  
.. [#fcn]
  Long, J., Shelhamer, E. and Darrell, T., 2015. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
  
.. [#fpn]
  Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B. and Belongie, S., 2017. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2117-2125).

.. [#lnkn]
  Chaurasia, A. and Culurciello, E., 2017, December. Linknet: Exploiting encoder representations for efficient semantic segmentation. In 2017 IEEE Visual Communications and Image Processing (VCIP) (pp. 1-4). IEEE.
  
.. [#psp]
  Zhao, H., Shi, J., Qi, X., Wang, X. and Jia, J., 2017. Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2881-2890).
  
.. [#dlv3]
  Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F. and Adam, H., 2018. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV) (pp. 801-818).
  
.. [#imn]
  Deng, J., Dong, W., Socher, R., Li, L.J., Li, K. and Fei-Fei, L., 2009, June. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition (pp. 248-255). Ieee.
  
.. [#dns]  
  Huang, G., Liu, Z., Van Der Maaten, L. and Weinberger, K.Q., 2017. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4700-4708).
  
.. [#eff]
  Tan, M. and Le, Q., 2019, May. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning (pp. 6105-6114). PMLR.
  
.. [#inc]
  Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z., 2016. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826).
  
.. [#mob]
  Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M. and Adam, H., 2017. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
  
.. [#res]
  He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
  
.. [#rsx]  
  Xie, S., Girshick, R., Dollár, P., Tu, Z. and He, K., 2017. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1492-1500).
  
.. [#sen]
  Hu, J., Shen, L. and Sun, G., 2018. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
  
.. [#vgg]
  Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
  
.. [#xcp]
  Chollet, F., 2017. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1251-1258).
  
.. [#fl]
  Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P., 2017. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
  
.. [#vnet]
  Milletari, F., Navab, N. and Ahmadi, S.A., 2016, October. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV) (pp. 565-571). IEEE.