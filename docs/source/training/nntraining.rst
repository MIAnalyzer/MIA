***********************
Neural Network Training
***********************

.. note::
  This manual is not intended as an introduction to deep learning. Please use some of the plethora of tutorials, books, original literature or other sources on leep learning.

The training of a neural network is the essential prerequisite to use the network as predictor. 

Simply put, a neural network is trained in a supervised manner by reducing the error between the network output and the ground truth, resulting in a change in the network weights so that the network would predict an output that resembles the ground truth.

The image labels (=ground truth) for the network training are different for each supported application (see :doc:`../applications/index`).


	
	
.. tip::
  * The question of how many labels are needed to train a neural network is not easy to answer, as it depends on the difficulty of the task and the error that can be tolerated as a result.  
  * An iterative training process is recommended:
 
    * Label some images as training images
    * Train a neural network with the training images
    * Predict some test images (that are not included in the training images)
    * Correct/Relabel the predicted images and include them in the training set
    * Retrain
    * Repeat until results are sufficient
	


