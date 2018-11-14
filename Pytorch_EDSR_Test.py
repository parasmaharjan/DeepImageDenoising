'''
########## 1024x1024 ##########

32 block EDSR with SE:
mean psnr: 29.57
mean ssim: 0.8889

16 block EDSR with SE:
mean psnr: 29.42
mean ssim: 0.8865

UNET Resulte:
mean psnr:  29.52
mean ssim:   0.8897

############ Full ############
32 block EDSR with SE:
mean psnr: 29.57
mean ssim: 0.8889

16 block EDSR with SE:
mean psnr: 29.42
mean ssim: 0.8865

UNET Resulte:
mean psnr:  29.52
mean ssim:   0.8897

'''

import os, time, scipy.io

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
import skimage.measure as skm
import cv2
from skimage import exposure


from EDSR import EDSR
import argparse

# Argument for EDSR
parser = argparse.ArgumentParser(description='EDSR')
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--scale', type=str, default=2,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=512,
                    help='output patch size')
parser.add_argument('--n_colors', type=int, default=4,
                    help='number of input color channels to use')
parser.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args = parser.parse_args()

torch.manual_seed(0)

input_dir = '../LearningToSeeInDark/dataset/Sony/short/'
gt_dir = '../LearningToSeeInDark/dataset/Sony/long/'
m_path = './checkpoint/EDSR/'
# m_path = './saved_model/edsr-se-ps-256-b-32/'
m_name = 'edsr_se_relu_ps_512_b_16_e5300.pth'
result_dir = './Final/i_edsr_se_relu_ps_512_b_16_e5300/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

ps = args.patch_size  # patch size for training

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


model = EDSR(args)
model.load_state_dict(torch.load(m_path + m_name))
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.cuda()

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

psnr = []
ssim = []
cnt = 0
with torch.no_grad():
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
            # input_full = input_full[:,:512, :512, :]

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # im = im[:1024,:1024]
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
            # scale_full = np.minimum(scale_full, 1.0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # im = im[:1024, :1024]
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            input_full = np.minimum(input_full, 1.0)

            in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).cuda()
            st = time.time()
            cnt +=1
            out_img = model(in_img)
            print('%d\tTime: %.3f'%(cnt, time.time()-st))

            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]
            scale_full = scale_full[0, :, :, :]
            origin_full = scale_full
            scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)  # scale the low-light image to the same mean of the groundtruth

            psnr.append(skm.compare_psnr(gt_full[:, :, :], output[:, :, :]))
            ssim.append(skm.compare_ssim(gt_full[:, :, :], output[:, :, :], multichannel=True))
            print('psnr: ', psnr[-1], 'ssim: ', ssim[-1])
            # temp = np.concatenate((scale_full_, gt_full_, output_), axis=1)
            # plt.clf()
            # plt.imshow(temp)
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)
            scipy.misc.toimage(origin_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%5d_00_%d_ori.png' % (test_id, ratio))
            scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%5d_00_%d_out.png' % (test_id, ratio))
            scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%5d_00_%d_scale.png' % (test_id, ratio))
            scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%5d_00_%d_gt.png' % (test_id, ratio))

print('mean psnr: ', np.mean(psnr))
print('mean ssim: ', np.mean(ssim))
print('done')
