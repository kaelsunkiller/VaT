#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   debuging.py    
@Contact :   kael.sunkiller@gmail.com
@License :   (C)Copyright 2020, Leozen-Yang

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
10/16/20 11:55 PM   Yang      0.0         None
"""

import numpy as np
import os
import six
from shutil import copy
import json
import random

# root = '/media/kirk/Projects/Self-critical.pytorch/data/cocotest_bu_fc'
# filelist = os.listdir(root)
# errorlist = []
# for file in filelist:
#     path = os.path.join(root, file)
#     x = open(path, 'rb').read()
#     try:
#         x = np.load(six.BytesIO(x))
#     except Exception as e:
#         print('error: {}'.format(path))
#         errorlist.append(file)
#         pass
#
# bk_root = '/media/kirk/Leozeen/Projects/Self-critical.pytorch/data/cocotest_bu_fc'
# for errorfile in errorlist:
#     bk_path = os.path.join(bk_root, errorfile)
#     cp_path = copy(bk_path, root)
#     print('copy {} to {}'.format(bk_path, cp_path))

data_path1 = '/Projects/Projects_ylz/self-critical.pytorch.2020.09/eval_results/.cache_vat_gmm32_ghm_nsc_test_n.json'
data_path2 = '/Projects/Projects_ylz/self-critical.pytorch.2020.09/eval_results/.cache_vat_gmm32_test_n.json'
data_path3 = '/Projects/Projects_ylz/self-critical.pytorch.2020.09/eval_results/.cache_transformer_nsc_test_n.json'
gt_path = '/Projects/Projects_ylz/self-critical.pytorch.2020.09/data/dataset_coco.json'
save_path = '/Projects/Projects_ylz/self-critical.pytorch.2020.09/vis/sample_vat_best'
source_img_path = '/Projects/Projects_ylz/COCO/data/val2014'
if not os.path.exists(save_path):
    os.makedirs(os.path.join(save_path, 'imgs'))
with open(data_path1, 'r') as reader:
    dic1 = json.load(reader)
with open(data_path2, 'r') as reader:
    dic2 = json.load(reader)
with open(data_path3, 'r') as reader:
    dic3 = json.load(reader)
with open(gt_path, 'r') as reader:
    gt = json.load(reader)
filelist = os.listdir('/Projects/Projects_ylz/COCO/data/test2014')
img2eval1 = dic1['div_stats']['ImgToEval']
img2eval2 = dic2['div_stats']['ImgToEval']
img2eval3 = dic3['div_stats']['ImgToEval']
imglist = img2eval1.keys()
samples = random.sample(imglist, 20)
gts = {str(_['cocoid']): {'sents': [s['raw'].lower() for s in _['sentences']], 'filename': _['filename']} for _ in gt['images']}
sample_dic = {}
best = 0
best_id = 0
difs = []
ids = []
for sample_id in imglist:
    mb2_1 = img2eval1[sample_id]['mBleu_2']
    cider_1 = np.mean([_['scores']['CIDEr'] for _ in img2eval1[sample_id]['individuals']])
    mb2_2 = img2eval2[sample_id]['mBleu_2']
    cider_2 = np.mean([_['scores']['CIDEr'] for _ in img2eval2[sample_id]['individuals']])
    dif_1 = cider_1 - mb2_1
    dif_2 = cider_2 - mb2_2
    # if dif > best and mb2 < 0.7:
    #     best = dif
    #     best_id = sample_id
    if cider_1 > cider_2 and cider_2 < 1 and mb2_1 < 0.7:
        difs.append(dif_1)
        ids.append(sample_id)
sample_ids = [ids[i] for i in np.argsort(difs)[-20:]]
best_difs = [difs[i] for i in np.argsort(difs)[-20:]]
print('search complete', best_difs)
# print(img2eval1[sample_ids[0]])
for sample_id in sample_ids:
    gtc = gts[sample_id]['sents']
    filename = gts[sample_id]['filename']
    captions1 = [_['caption'] for _ in img2eval1[sample_id]['individuals']]
    captions2 = [_['caption'] for _ in img2eval2[sample_id]['individuals']]
    captions3 = [_['caption'] for _ in img2eval3[sample_id]['individuals']]
    sample_dic[sample_id] = {'gts': gtc,
                             'vat': {'captions': captions1, 'indiv': img2eval1[sample_id]},
                             'diverse': {'captions': captions2, 'indiv': img2eval2[sample_id]},
                             'accurate': {'captions': captions3, 'indiv': img2eval3[sample_id]}
                             }
    source_file = os.path.join(source_img_path, filename)
    target_dir = os.path.join(save_path)
    copy(source_file, target_dir)
with open(os.path.join(save_path, 'captions_vat.json'), 'w') as writter:
    json.dump(sample_dic, writter)
print('finished')

# with open('/media/kirk/DATA/iu_xray/annotation.json', 'r') as reader:
#     dic1 = json.load(reader)
#     print(1)
# import sys
# sys.path.append("coco-caption")


# from pycocotools.coco import COCO
#
# dataDir = '/Projects/Projects_ylz/COCO/data'
# dataType = 'train2014'
# # dataType='train2017'
# annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
#
# # initialize COCO api for instance annotations
# coco = COCO(annFile)
#
# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# cat_nms = [cat['name'] for cat in cats]
# print('number of categories: ', len(cat_nms))
# print('COCO categories: \n', cat_nms)
#
# # 统计各类的图片数量和标注框数量
# ImgIds = {}
# for cat_name in cat_nms:
#     catId = coco.getCatIds(catNms=cat_name)  # 1~90
#     imgId = coco.getImgIds(catIds=catId)  # 图片的id
#     annId = coco.getAnnIds(catIds=catId)  # 标注框的id
#     for imgid in imgId:
#         if imgid in ImgIds:
#             ImgIds[imgid] = ImgIds[imgid] + [cat_name]
#             assert ImgIds[imgid] is not None, ImgIds[imgid]
#         else:
#             ImgIds[imgid] = [cat_name]
#             assert ImgIds[imgid] is not None, ImgIds[imgid]
#
#         assert ImgIds[imgid] is not None
#         # if ImgIds[imgid] is None:
#         #     ImgIds[imgid] = [cat_name]
#
#             # print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))
# catImgcop = {}
# # catCombines = set(ImgIds.values())
# # catCombines = dict([catCombines, list(range(len(catCombines)))])
# for key, value in ImgIds.items():
#     value = ','.join(value)
#     catImgcop[value] = catImgcop[value] + [key] if value in catImgcop else [key]
#     if catImgcop[value] is None:
#         catImgcop[value] = [key]
# print('categories-image pairs number:', len(catImgcop))
# with open('catImgCombines.txt', 'w') as writer:
#     for key, value in catImgcop.items():
#         # print("{} {:<10d}".format(key, len(value)))
#         writer.write("{}:{:<10d}\n".format(key, len(value)))
