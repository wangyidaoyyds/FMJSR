# encoding: utf-8

from pathlib import Path
from easydict import EasyDict as edict

def get_config(MatName = 'AR_79', dctOrDeep = 'Deep'):
    conf = edict()
    conf.MatName = 'save_canshu/feature/{}.mat'.format(MatName)

    if MatName == 'GT_50':
        conf.LabelNum = 50
        conf.GalleryLen = 350
    elif MatName == 'EAR3_50':
        conf.LabelNum = 50
        conf.GalleryLen = 350
    elif MatName == 'AR_79':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName == 'AR_100':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName == 'EAR3_79' or MatName == 'EAR3_79_13':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName == 'PolyU1':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName == '79AR_77AR':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName == '79AR_GT':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName == '100AR_GT':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName == 'EAR79_EAR50':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName == 'EAR3_77EAR':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName == '100PolyU1_50PolyU1':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName == 'PolyU1_PolyU2':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName == 'PolyU_400':
        conf.LabelNum = 400
        conf.GalleryLen = 2800
    return conf

def get_config_spoof_multimodal(MatName_1 = 'AR_79', MatName_2 = 'EAR3_79', dctOrDeep = 'deep'):
    conf = edict()
    conf.MatName_1 = 'save_canshu/feature/{}.mat'.format(MatName_1)
    conf.MatName_2 = 'save_canshu/feature/{}.mat'.format(MatName_2)

    if MatName_1 == 'AR_79':
        conf.LabelNum = 77
        conf.GalleryLen = 77 * 7

    elif MatName_1 == 'AR_100':
        conf.LabelNum = 100
        conf.GalleryLen = 700

    elif MatName_1 == '79AR_GT':
        conf.LabelNum = 50
        conf.GalleryLen = 350
    elif MatName_1 == '100AR_GT':
        conf.LabelNum = 50
        conf.GalleryLen = 350

    return conf

def get_config_multimodal(MatName_1 = 'AR_79', MatName_2 = 'EAR3_79', dctOrDeep = 'deep'):
    conf = edict()
    print(MatName_1)
    print(MatName_2)
    conf.MatName_1 = 'save_canshu/feature/{}.mat'.format(MatName_1)
    conf.MatName_2 = 'save_canshu/feature/{}.mat'.format(MatName_2)

    if MatName_1 == 'GT_50':
        conf.LabelNum = 50
        conf.GalleryLen = 350
    elif MatName_1 == 'EAR3_50':
        conf.LabelNum = 50
        conf.GalleryLen = 350
    elif MatName_1 == 'AR_79' or MatName_1 == '79AR_OCC79AR':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName_1 == 'AR_100' or MatName_1 == '100AR_OCC100AR':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName_1 == 'EAR3_79':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName_1 == 'PolyU1':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName_1 == '79AR_77AR':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName_1 == '79AR_GT':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName_1 == '100AR_GT':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName_1 == 'EAR79_EAR50':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName_1 == 'EAR3_77EAR':
        conf.LabelNum = 79
        conf.GalleryLen = 553
    elif MatName_1 == '100PolyU1_20PolyU1':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    elif MatName_1 == 'PolyU1_PolyU2':
        conf.LabelNum = 100
        conf.GalleryLen = 700
    return conf