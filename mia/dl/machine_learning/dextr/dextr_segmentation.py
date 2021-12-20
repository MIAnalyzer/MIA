

# adapted from https://github.com/scaelles/DEXTR-KerasTensorflow
# licence GPLv3, https://github.com/scaelles/DEXTR-KerasTensorflow/blob/master/LICENSE

import numpy as np
from dl.machine_learning.dextr.dextr import DEXTR
import dl.machine_learning.dextr.helpers as helpers
from dl.utils.dl_downloads import get_dextr

class DEXTR_Segmentation():
    def __init__(self):
        self.pad = 50
        self.thres = 0.8
        self.modelName = None 
        self.net = None
        
    def loadModel(self):
        try:
            if self.net is None:
                if self.modelName is None:
                    self.modelName =  get_dextr()
                self.net = DEXTR(nb_classes=1, resnet_layers=101, input_shape=(512, 512), weights=self.modelName, num_input_channels=4, classifier='psp', sigmoid=True)
        except:
            # can not load file
            self.net = None

    def predict(self, image, extreme_points_ori):
        if not self.net:
            self.loadModel()
            if not self.net:
                return 

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=self.pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [self.pad, self.pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)

        # Run a forward pass
        pred = self.net.model.predict(input_dextr[np.newaxis, ...])[0, :, :, 0]
        result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=self.pad) > self.thres
        return result.astype(np.int)