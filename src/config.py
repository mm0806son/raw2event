#! python3
# -*- encoding: utf-8 -*-

from easydict import EasyDict as edict

__C     = edict()
cfg     = __C

# SENSOR
__C.SENSOR                              = edict()
__C.SENSOR.CAMERA_TYPE = 'Raw2DVS346'

__C.SENSOR.K = None
if cfg.SENSOR.CAMERA_TYPE == 'DVS346':
    __C.SENSOR.K = [0.00018 * 29250, 20, 0.0001, 1e-7, 5e-9, 0.00001]
elif cfg.SENSOR.CAMERA_TYPE == 'DVS240':
    __C.SENSOR.K = [0.000094 * 47065, 23, 0.0002, 1e-7, 5e-8, 0.00001]
elif cfg.SENSOR.CAMERA_TYPE == 'Raw2DVS346':
    __C.SENSOR.K = [2.388, 4.166e-7, 1.541e-6, 9.768e-8, 1.466e-11, 9.824e-6]
elif cfg.SENSOR.CAMERA_TYPE == 'RGB2DVS346':
    __C.SENSOR.K = [5.332474147628972,0.9003332027266823,8.288263352543993e-06,1.0992397172828087e-07,3.302652963977818e-09,1.1012444038716504e-07]



# Directories
__C.DIR                                 = edict()
__C.DIR.IN_PATH = 'data_samples/interp/'
__C.DIR.OUT_PATH = 'data_samples/output/'


# Visualize
__C.Visual                              = edict()
__C.Visual.FRAME_STEP = 5
