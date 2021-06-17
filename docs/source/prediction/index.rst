.. include:: ../icons.txt

**********
Prediction
**********

Prediction or inference is the process of using a trained neural network to predict unknown images. Model weights are not updated during prediction.

.. tip::
  Multiple predictions of a neural network for the same input image lead to identical results. You can always reproduce results by saving and keeping track of your trained models.
  
Press |predict| **Predict** button to predict to make a prediction for the current image with the current model. Prediction also works during training when you want to get a quick snapshot of the model's prediction output.

.. note::
  The prediction is made for the current field of view. Everything outside the field of view is unaffected.
  
Press the |predictall| **Predict All** button to predict all images in the test folder, no subfolders included. Note that the prediction might take some time. The prediction is run in a background thread, you can continue working with MIA during prediction. 
The :ref:`status bar <statusbar>` shows the progress of the prediction.

.. caution::
  Please be aware that it is not recommended to change the labels of images that are currently predicted, as this might lead to conflicts.


  
