import os
import glob
import numpy as np
import scipy.io as scio
from scipy.io import loadmat ,savemat
import argparse
import yaml
import torch
import utils.tools
import model.optics_rgb
import math
import matplotlib.pyplot as plt

def config(path):
    """ Config file for training.
    """
    # Config file
    current_path = os.getcwd()
    with open(path, encoding='utf-8') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    lens = args.get('lens', args['filename'])
    args['in_path']= os.path.join(current_path,args['input_dir'],lens+'.xlsx')
    list = ['mat', 'npy', 'crop']

    for key in list:
        path = os.path.join(args['dataset'],args['filename'], key)
        args[key] = path
        if not os.path.exists(path):
            os.makedirs(path)
    return args

def save_npy(gt_list,weights, filename_l, path):
    lens = len(gt_list)
    width = (len(gt_list[0]['sfr'][:, 1]) + 1) // 2

    print(width)

    for img_idx in range(lens):
        filename = os.path.basename(filename_l[img_idx])
        npy_path = os.path.join(path,filename[:-4])
        # red channel(0), green channel(1), blue channel(2)
        sfr= gt_list[img_idx]['sfr'][:,1:4][0:width]
        # sfr_g = gt_list[img_idx]['sfr'][:,2][0:width]
        # sfr_b = gt_list[img_idx]['sfr'][:,3][0:width]
        weight = weights[img_idx]
        rot = gt_list[img_idx]['rot']
        fov = gt_list[img_idx]['fov']
        offset = gt_list[img_idx]['offset']
        # if fov > 35:
        #     print(rot)
        #     print(offset)
        # dict ={'sfr':sfr,'weight':weight,'rot':rot,'fov':fov}
        np.savez(npy_path,sfr=sfr,weight=weight,rot=rot,fov=fov,offset=offset)


def fov_from_position(temp, efl, pixelsize):
    """ 实拍数据:由边缘中心像素坐标 + 焦距/像元尺寸计算视场角。
    r = 边缘中心到图像中心的像素距离, fov = atan(r * pixelsize / efl)。 """
    cx, cy = float(temp['cx'][0, 0]), float(temp['cy'][0, 0])
    dx = cx - float(temp['img_w'][0, 0]) / 2
    dy = cy - float(temp['img_h'][0, 0]) / 2
    r = math.sqrt(dx ** 2 + dy ** 2)
    fov = math.degrees(math.atan(r * pixelsize / efl))
    return np.array([[fov]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/ss.yaml')
    ns = parser.parse_args()
    path = ns.config
    args = config(path)
    print(args['npy'])
    real = args.get('real', False)
    if real:
        efl = args.get('efl')
        pixelsize = args.get('pixelsize')
        hfov = int(args.get('hfov', args.get('hfov_max', 35))) + 1
    else:
        IS = model.optics_rgb.IS(filepath=args['in_path'],
                                 efl=args.get('efl'), pixelsize=args.get('pixelsize'))
        efl = pixelsize = None
        hfov = int(IS.hfov) + 1
    if real:
        directory = args.get('mat_dir')
        if not directory:
            raise SystemExit('real 模式必须在配置中指定 mat_dir(实拍 .mat 目录)')
    else:
        directory = args.get('mat_dir', os.path.join(args['dataset'], args['filename'], 'mat', args.get('noise', '')))

    mat_files = sorted([file for file in os.listdir(directory) if file.endswith('.mat')])
    data, filename = [], []
    for file_name in mat_files:
        temp = loadmat(os.path.join(directory, file_name))
        item = {'rot': temp['rot'] + 360 if temp['rot'] < 0 else temp['rot'],
                'sfr': temp['sfr'],
                'fov': temp['fov'],
                'offset': temp['offset']}
        if real:
            item['fov'] = fov_from_position(temp, efl, pixelsize)
        data.append(item)
        filename.append(file_name)

    'determine weight matrix'
    fov_m, weights_all, gt_list ,filename_l = [], [], [],[]
    for i in range(hfov-1):
        # filter all the mtf belong [i,i+1)
        filtered_list = [item for item in data if i <= item['fov'] < i + 1]
        filtered_list = sorted(filtered_list, key=lambda x: x['rot'])
        sel_file = [filename[index] for index, item in enumerate(data) if i <= item['fov'] < i + 1]
        filename_l.append(sel_file)
        gt_list.append(filtered_list)
        fov_m.append(len(filtered_list))

        angles = np.array([item['rot'] for item in filtered_list], dtype=np.float64)

        angles = torch.tensor(angles)
        # fov_weight = math.sin(math.radians((i * 80 / hfov + 10)))
        fov_weight = 1
        weights_m = utils.tools.weights(angles, bins=8)
        weights_all.append(weights_m*fov_weight)


    weights_all1 = torch.cat(weights_all)
    weights_all1 = weights_all1/torch.sum(weights_all1)
    gt_list = [item for sublist in gt_list for item in sublist]
    filename_l = [item for sublist in filename_l for item in sublist]
    save_npy(gt_list, weights_all1, filename_l, args['npy'])









