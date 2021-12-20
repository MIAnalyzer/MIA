
from dl.utils.dl_downloads import get_hed
import cv2
# see https://github.com/opencv/opencv/blob/master/samples/dnn/edge_detection.py


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0
 
    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
 
        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width
 
        return [[batchSize, numChannels, height, width]]
 
    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]


class HED_Segmentation():
    def __init__(self):
        # crashs when double register
        cv2.dnn_unregisterLayer('Crop')
        cv2.dnn_registerLayer('Crop', CropLayer)
        self.proto = './dl/machine_learning/deploy.prototxt'
        self.model = None 
        self.net = None

    def applyHED(self, image):
        try:
            if self.net is None:
                if self.model is None:
                    self.model = get_hed()
                self.net = cv2.dnn.readNet(self.proto, self.model)
        except:
            # can not load file
            self.net = None

        if not self.net:
            raise # model not loaded
        if image.dtype == 'uint16':
            image = (image/256).astype('uint8')
            
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR )
        
        orig_height, orig_width = image.shape[:2]
        while True:
            height, width = image.shape[:2]

            blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height),mean=(104.00698793, 116.66876762, 122.67891434),swapRB=False, crop=False)
            self.net.setInput(blob)
            try:
                hed = self.net.forward()
                break
            except:
                pass
            
            if width*height  < 100:
                raise
            image = cv2.resize(image, (int(width*0.9), int(height*0.9)))
             
        hed = cv2.resize(hed[0, 0], (orig_width, orig_height))
        hed = (255 * hed).astype("uint8")
        return hed

