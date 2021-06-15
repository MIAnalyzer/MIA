.. include:: ../icons.txt

**********
Train Data
**********

The Training Data Window opens automatically after the training is started.

.. figure:: images/trainplot.png
  :class: shadow-image
  :align: center
  
  The Training Data Window
  
The Training Data Window keeps track of the training process. 

The top row shows the training loss and training metric data per iteration.

The bottom row shows the validation loss and validation metric (red) and training loss and training metric (dashed blue).

Press **Stop Training** to stop the training process.

You can use the **Resume Training** button to resume training.

.. note::
  There are 2 options to continue training:
  
  * **Resume Training**: The training is resumed and continued at the last epoch. The training process is not reinitialized. Changes to training parameters have no effect (batch size or epochs might be changed).
  * **Start Training** (see :doc:`./nntraining`): The training is continued (already trained weights of the model are preserved!). All data is reloaded and all training settings can be adjusted, except settings that influence model architecture.
  
.. tip::
  * Resume training can be used for fast pause/resuming. The training process is continued as is, changes to training settings or training data have no effect. 
  * Whenever you want to add data to the training set or change training settings use |train| **Start Training**. All data is reloaded, which takes additional time.
  * To reset a model completely (including model weights), for example to change the model architecture, press |clear| **Reset Model** in the Main Window. All training progress is lost!