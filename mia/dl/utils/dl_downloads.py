# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:32:17 2021

@author: koerber
"""


### to do:
# rework and unify
    
import os
from tensorflow.python.keras.utils.data_utils import get_file

SUBDIR = 'models'
MIAURL = 'https://github.com/MIAnalyzer/MIA/releases/download/weights/'


def startFunction():
    return True

def failFunction():
    return

# quick and dirty callback for gui
callBackstart = startFunction
callBackfail = failFunction

def checkFile(fname):
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, SUBDIR)
    fpath = os.path.join(datadir, fname)
    if os.path.exists(fpath):
        return True
    return False

def getTargetFile(fname, url, fhash):
    if checkFile(fname):
        return get_file(fname, url, file_hash = fhash, cache_subdir=SUBDIR)
    else:
        if callBackstart():
            try:
                file = get_file(fname, url, file_hash = fhash, cache_subdir=SUBDIR)
                return file
            except:
                callBackfail()

def get_hed():
    fname = 'hed_pretrained_bsds.caffemodel'
    url = MIAURL + fname
    # original link not valid atm
    # url = 'http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'
    fhash = '8d371aa3c27a5b2a31228282a047965d'
    return getTargetFile(fname, url, fhash)

def get_dextr():
    fname = 'dextr_coco.h5'
    url = MIAURL + fname
    fhash = '16415d65f6e15d3fa053afa21d5927e0'
    return getTargetFile(fname, url, fhash)

def get_deeplabX():
    fname = 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5'
    url = MIAURL + fname
    fhash = "b979702082524aca6a249fd546f8ea13"
    return getTargetFile(fname, url, fhash)
    
def get_deeplabM():
    fname = 'deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5'
    url= MIAURL + fname
    fhash = "370affe402895f37d171fb7123eac200"
    return getTargetFile(fname, url, fhash)

def get_EfficientNet(model_name, include_top):
    hashes = {
        'efficientnet-b0': ('163292582f1c6eaca8e7dc7b51b01c61'
                            '5b0dbc0039699b4dcd0b975cc21533dc',
                            'c1421ad80a9fc67c2cc4000f666aa507'
                            '89ce39eedb4e06d531b0c593890ccff3'),
        'efficientnet-b1': ('d0a71ddf51ef7a0ca425bab32b7fa7f1'
                            '6043ee598ecee73fc674d9560c8f09b0',
                            '75de265d03ac52fa74f2f510455ba64f'
                            '9c7c5fd96dc923cd4bfefa3d680c4b68'),
        'efficientnet-b2': ('bb5451507a6418a574534aa76a91b106'
                            'f6b605f3b5dde0b21055694319853086',
                            '433b60584fafba1ea3de07443b74cfd3'
                            '2ce004a012020b07ef69e22ba8669333'),
        'efficientnet-b3': ('03f1fba367f070bd2545f081cfa7f3e7'
                            '6f5e1aa3b6f4db700f00552901e75ab9',
                            'c5d42eb6cfae8567b418ad3845cfd63a'
                            'a48b87f1bd5df8658a49375a9f3135c7'),
        'efficientnet-b4': ('98852de93f74d9833c8640474b2c698d'
                            'b45ec60690c75b3bacb1845e907bf94f',
                            '7942c1407ff1feb34113995864970cd4'
                            'd9d91ea64877e8d9c38b6c1e0767c411'),
        'efficientnet-b5': ('30172f1d45f9b8a41352d4219bf930ee'
                            '3339025fd26ab314a817ba8918fefc7d',
                            '9d197bc2bfe29165c10a2af8c2ebc675'
                            '07f5d70456f09e584c71b822941b1952'),
        'efficientnet-b6': ('f5270466747753485a082092ac9939ca'
                            'a546eb3f09edca6d6fff842cad938720',
                            '1d0923bb038f2f8060faaf0a0449db4b'
                            '96549a881747b7c7678724ac79f427ed'),
        'efficientnet-b7': ('876a41319980638fa597acbbf956a82d'
                            '10819531ff2dcb1a52277f10c7aefa1a',
                            '60b56ff3a8daccc8d96edfd40b204c11'
                            '3e51748da657afd58034d54d3cec2bac')
    }
    
    if include_top:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
        file_hash = hashes[model_name][0]
    else:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
        file_hash = hashes[model_name][1]
    
    return getTargetFile(file_name, MIAURL + file_name, file_hash)


def get_inception_resnet_v2(include_top):
    if include_top:
        fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
        file_hash='e693bd0210a403b3192acc6073ad2e96'
    else:
        fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash='d19885ff4a710c122648d3b5c3b684e4'
    return getTargetFile(fname, MIAURL + fname, file_hash)

def get_inceptionv3(include_top):
    if include_top:
        fname = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
        url = MIAURL + fname
        fhash = '9a0d58056eeedaa3f26cb7ebd46da564'
    else:
        fname = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        url = MIAURL + fname
        fhash = 'bcbd6486424b2319ff4ef7d526e38f63'
    return getTargetFile(fname, url, fhash) 

def get_xception(include_top):
    if include_top:
        fname = 'xception_weights_tf_dim_ordering_tf_kernels.h5'
        url = MIAURL + fname
        fhash = '0a58e3b7378bc2990ea3b43d5981f1f6'
    else:
        fname = 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        url = MIAURL + fname
        fhash = 'b0042744bf5b25fce3cb969f33bebb97'
    return getTargetFile(fname, url, fhash)
        
def get_NasNetLarge(include_top):
    url = ('https://github.com/titu1994/Keras-NASNet/'
                         'releases/download/v1.2/')
    if include_top:
        fname = 'NASNet-large.h5'
        fhash = '11577c9a518f0070763c2b964a382f17'
    else:
        fname = 'NASNet-large-no-top.h5'
        fhash = 'd81d89dc07e6e56530c4e77faddd61b5'

    return getTargetFile(fname, MIAURL+fname, fhash)
    
    
def get_NasNetMobile(include_top):
    if include_top:
        fname = 'NASNet-mobile.h5'
        fhash = '020fb642bf7360b370c678b08e0adf61'
    else:
        fname = 'NASNet-mobile-no-top.h5'
        fhash = '1ed92395b5b598bdda52abe5c0dbfd63'
    
    return getTargetFile(fname, MIAURL+fname, fhash)



def get_MobileNet(include_top, alpha, rows):
    if alpha != 1.0 or rows != 224:
        raise 'not supported'

    if include_top:
        fname = 'mobilenet_1_0_224_tf.h5'
        fhash = '03394917f9a9ea3362f46332a5b6a215'
        
    else:
        fname = 'mobilenet_1_0_224_tf_no_top.h5'
        fhash = '725ccbd03d61d7ced5b5c4cd17e7d527'
        
    return getTargetFile(fname, MIAURL+fname, fhash)

def get_MobileNetv2(include_top, alpha, rows):    
    if alpha != 1.0 or rows != 224:
        raise 'not supported'

    if include_top:
        fname = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
        fhash = '321911b381f6fd09b744356f6796ee18'
        
    else:
        fname = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
        fhash = 'f1a68526548f7541cda07e19ba6c85f4'
        
    return getTargetFile(fname, MIAURL+fname, fhash)


def get_DenseNet(include_top, blocks):    
    if include_top:
        if blocks == [6, 12, 24, 16]:
            fname = 'densenet121_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash='9d60b8095a5708f2dcce2bca79d332c7'
        elif blocks == [6, 12, 32, 32]:
            fname = 'densenet169_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash='d699b8f76981ab1b30698df4c175e90b'
        elif blocks == [6, 12, 48, 32]:
            fname = 'densenet201_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807'
    else:
        if blocks == [6, 12, 24, 16]:
            fname = 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash='30ee3e1110167f948a6b9946edeeb738'
        elif blocks == [6, 12, 32, 32]:
            fname = 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash='b8c4d4c20dd625c148057b9ff1c1176b'
        elif blocks == [6, 12, 48, 32]:
            fname = 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash='c13680b51ded0fb44dff2d8f86ac8bb1'
    
    return getTargetFile(fname, MIAURL + fname, file_hash) 
    

def get_ResNet(include_top, model_name):
    hashes = {
        'resnet50': ('2cb95161c43110f7111970584f804107',
                     '4d473c1dd8becc155b73f8504c6f6626'),
        'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                      '88cf7a10940856eca736dc7b7e228a21'),
        'resnet152': ('100835be76be38e30d865e96f2aaae62',
                      'ee4c566cf9a93f14d82f913c2dc6dd0c'),
        'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                       'fac2f116257151a9d068a22e544a4917'),
        'resnet101v2': ('6343647c601c52e1368623803854d971',
                        'c0ed64b8031c3730f411d2eb4eea35b5'),
        'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                        'ed17cf2e0169df9d443503ef94b23b33'),
        'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                      '62527c363bdd9ec598bed41947b379fc'),
        'resnext101': ('34fb605428fcc7aa4d62f44404c11509',
                       '0f678c91647380debd923963594981b3')
    }
    
    if include_top:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
        file_hash = hashes[model_name][0]
    else:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash = hashes[model_name][1]
    return getTargetFile(file_name,  MIAURL + file_name,  file_hash)
    
    
    
def get_vgg16(include_top):  
    if include_top:
        fname = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        url = MIAURL + fname
        file_hash='64373286793e3c8b2b4e3219cbf3544b'
        

    else:
        fname = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        url = MIAURL + fname 
        file_hash = '6d6bbae143d832006294945121d1f1fc'
        
    return getTargetFile(fname,  url, file_hash)

def get_vgg19(include_top):  
    if include_top:
        fname = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
        url = MIAURL + fname 
        file_hash='cbe5617147190e668d6c5d5026f83318'
        
    else:
        fname = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        url = MIAURL + fname 
        file_hash = '253f8cb515780f3b799900260a226db6'
        
    return getTargetFile(fname,  url,  file_hash)




WEIGHTS_COLLECTION = [

    # ResNet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'resnet18_imagenet_1000.h5',
        'name': 'resnet18_imagenet_1000.h5',
        'md5': '64da73012bb70e16c901316c201d9803',
    },

    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'resnet18_imagenet_1000_no_top.h5',
        'name': 'resnet18_imagenet_1000_no_top.h5',
        'md5': '318e3ac0cd98d51e917526c9f62f0b50',
    },

    # ResNet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'resnet34_imagenet_1000.h5',
        'name': 'resnet34_imagenet_1000.h5',
        'md5': '2ac8277412f65e5d047f255bcbd10383',
    },

    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'resnet34_imagenet_1000_no_top.h5',
        'name': 'resnet34_imagenet_1000_no_top.h5',
        'md5': '8caaa0ad39d927cb8ba5385bf945d582',
    },

    # ResNet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'resnet50_imagenet_1000.h5',
        'name': 'resnet50_imagenet_1000.h5',
        'md5': 'd0feba4fc650e68ac8c19166ee1ba87f',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'resnet50_imagenet_1000_no_top.h5',
        'name': 'resnet50_imagenet_1000_no_top.h5',
        'md5': 'db3b217156506944570ac220086f09b6',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet11k-places365ch',
        'classes': 11586,
        'include_top': True,
        'url': MIAURL + 'resnet50_imagenet11k-places365ch_11586.h5',
        'name': 'resnet50_imagenet11k-places365ch_11586.h5',
        'md5': 'bb8963db145bc9906452b3d9c9917275',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet11k-places365ch',
        'classes': 11586,
        'include_top': False,
        'url': MIAURL + 'resnet50_imagenet11k-places365ch_11586_no_top.h5',
        'name': 'resnet50_imagenet11k-places365ch_11586_no_top.h5',
        'md5': 'd8bf4e7ea082d9d43e37644da217324a',
    },

    # ResNet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'resnet101_imagenet_1000.h5',
        'name': 'resnet101_imagenet_1000.h5',
        'md5': '9489ed2d5d0037538134c880167622ad',
    },

    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'resnet101_imagenet_1000_no_top.h5',
        'name': 'resnet101_imagenet_1000_no_top.h5',
        'md5': '1016e7663980d5597a4e224d915c342d',
    },

    # ResNet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'resnet152_imagenet_1000.h5',
        'name': 'resnet152_imagenet_1000.h5',
        'md5': '1efffbcc0708fb0d46a9d096ae14f905',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'resnet152_imagenet_1000_no_top.h5',
        'name': 'resnet152_imagenet_1000_no_top.h5',
        'md5': '5867b94098df4640918941115db93734',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet11k',
        'classes': 11221,
        'include_top': True,
        'url': MIAURL + 'resnet152_imagenet11k_11221.h5',
        'name': 'resnet152_imagenet11k_11221.h5',
        'md5': '24791790f6ef32f274430ce4a2ffee5d',
    },

    {
        'model': 'resnet152',
        'dataset': 'imagenet11k',
        'classes': 11221,
        'include_top': False,
        'url': MIAURL + 'resnet152_imagenet11k_11221_no_top.h5',
        'name': 'resnet152_imagenet11k_11221_no_top.h5',
        'md5': '25ab66dec217cb774a27d0f3659cafb3',
    },

    # ResNeXt50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'resnext50_imagenet_1000.h5',
        'name': 'resnext50_imagenet_1000.h5',
        'md5': '7c5c40381efb044a8dea5287ab2c83db',
    },

    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'resnext50_imagenet_1000_no_top.h5',
        'name': 'resnext50_imagenet_1000_no_top.h5',
        'md5': '7ade5c8aac9194af79b1724229bdaa50',
    },

    # ResNeXt101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'resnext101_imagenet_1000.h5',
        'name': 'resnext101_imagenet_1000.h5',
        'md5': '432536e85ee811568a0851c328182735',
    },

    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'resnext101_imagenet_1000_no_top.h5',
        'name': 'resnext101_imagenet_1000_no_top.h5',
        'md5': '91fe0126320e49f6ee607a0719828c7e',
    },

    # SE models
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'seresnet50_imagenet_1000.h5',
        'name': 'seresnet50_imagenet_1000.h5',
        'md5': 'ff0ce1ed5accaad05d113ecef2d29149',
    },

    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'seresnet50_imagenet_1000_no_top.h5',
        'name': 'seresnet50_imagenet_1000_no_top.h5',
        'md5': '043777781b0d5ca756474d60bf115ef1',
    },

    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'seresnet101_imagenet_1000.h5',
        'name': 'seresnet101_imagenet_1000.h5',
        'md5': '5c31adee48c82a66a32dee3d442f5be8',
    },

    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'seresnet101_imagenet_1000_no_top.h5',
        'name': 'seresnet101_imagenet_1000_no_top.h5',
        'md5': '1c373b0c196918713da86951d1239007',
    },

    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'seresnet152_imagenet_1000.h5',
        'name': 'seresnet152_imagenet_1000.h5',
        'md5': '96fc14e3a939d4627b0174a0e80c7371',
    },

    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'seresnet152_imagenet_1000_no_top.h5',
        'name': 'seresnet152_imagenet_1000_no_top.h5',
        'md5': 'f58d4c1a511c7445ab9a2c2b83ee4e7b',
    },

    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'seresnext50_imagenet_1000.h5',
        'name': 'seresnext50_imagenet_1000.h5',
        'md5': '5310dcd58ed573aecdab99f8df1121d5',
    },

    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'seresnext50_imagenet_1000_no_top.h5',
        'name': 'seresnext50_imagenet_1000_no_top.h5',
        'md5': 'b0f23d2e1cd406d67335fb92d85cc279',
    },

    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'seresnext101_imagenet_1000.h5',
        'name': 'seresnext101_imagenet_1000.h5',
        'md5': 'be5b26b697a0f7f11efaa1bb6272fc84',
    },

    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'seresnext101_imagenet_1000_no_top.h5',
        'name': 'seresnext101_imagenet_1000_no_top.h5',
        'md5': 'e48708cbe40071cc3356016c37f6c9c7',
    },

    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'senet154_imagenet_1000.h5',
        'name': 'senet154_imagenet_1000.h5',
        'md5': 'c8eac0e1940ea4d8a2e0b2eb0cdf4e75',
    },

    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'senet154_imagenet_1000_no_top.h5',
        'name': 'senet154_imagenet_1000_no_top.h5',
        'md5': 'd854ff2cd7e6a87b05a8124cd283e0f2',
    },

    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'seresnet18_imagenet_1000.h5',
        'name': 'seresnet18_imagenet_1000.h5',
        'md5': '9a925fd96d050dbf7cc4c54aabfcf749',
    },

    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'seresnet18_imagenet_1000_no_top.h5',
        'name': 'seresnet18_imagenet_1000_no_top.h5',
        'md5': 'a46e5cd4114ac946ecdc58741e8d92ea',
    },

    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': MIAURL + 'seresnet34_imagenet_1000.h5',
        'name': 'seresnet34_imagenet_1000.h5',
        'md5': '863976b3bd439ff0cc05c91821218a6b',
    },

    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': MIAURL + 'seresnet34_imagenet_1000_no_top.h5',
        'name': 'seresnet34_imagenet_1000_no_top.h5',
        'md5': '3348fd049f1f9ad307c070ff2b6ec4cb',
    },

]

