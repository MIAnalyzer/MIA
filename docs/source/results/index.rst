.. include:: ../icons.txt

*******
Results
*******

To save export the results open the results window by pressing |results| **Results**.

.. figure:: images/results.png
  :class: shadow-image
  :align: center
  
  The results window
  
**Use scaled size** can be checked to scale all values from pixel to a specified distance.

To set the scale in **pixel per mm** fill in the corresponding field or use the |measure| tool.
When selecting the measure tool, you can set 2 markers by pressing :kbd:`left_mouse` in the image to measure the distance between the 2 markers and and align them to a specified distance. 
The scale in pixel per mm is automatically filled in the corresponding field.

.. figure:: images/setscale.png
  :class: shadow-image
  :align: center
  
  Set the scale by measuring the distance defined by the scale bar
  
Press |Settings| **Settings** to open the settings menu, specific for each :doc:`application <../applications/index>`. In the settings menu you can specify the save options.

To export labels as masks, for example to use with another software, press |exportmask| **Export Mask** and select a location. To export all labels in the test :ref:`test folder <folders>` press |exportallmasks| **Export all Masks**, the corresponding masks are saved automatically in the subfolder named 'exported_masks' present in the current test folder.

.. note::
  Masks are saved as tiff-files, having the same size as the corresponding image, with all pixel that are labelled as background set to 255 and all pixel that are labelled by a class set to the value corresponding to that class (i.e. first class set to 1).

Press |savemodel| **Save** to save the results of the currently displayed image.

Press |saveall| **Save all** to save the reuslts of all images in the test folder.

