import cv2
import os
import sys
sys.path.append("/Users/dredonieto/Documents/git/keras-retinanet")

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

class Detector():
    ''' Abstract detector class 
    '''
    def __init__(
        self, 
        model_path = None,
        image_path = None):

        '''
            Initialize Detector object 
        '''
        if not os.path.exists(model_path):
            raise ValueError('The `model_path` argument should be '
                             'a valid path')
        else:
            self.model_path = model_path
        
        if not os.path.exists(image_path):
            raise ValueError('The `image_path` argument should be'
                             'a valid path')
        else:
            self.image_path = image_path

        self.scale = None
        self.model = None

    def detect(self):
        ''' This method will be used to detect objects
        '''
        pass

    def _read_and_preprocess(self):
        ''' This function is used to load and preprocess the image 

        '''
        image = read_image_bgr(self.image_path)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        return image, scale


class BibDetector(Detector):
    '''
        Bib detector class
    '''

    def __init__(self, **kwargs):
        ''' Initialize BibDetector
            
        '''
        super(BibDetector, self).__init__(**kwargs)