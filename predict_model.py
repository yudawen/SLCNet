from torch import nn
import torch
import warnings
import os
from torch.autograd import Variable
import numpy as np
import cv2 as cv

from torch.nn import functional as F
import gdal, shutil
import imageio

from tools import LoadTest
import cv2

warnings.filterwarnings('ignore')
from SLCNet import SLCNet

def tianchong_xy(img, need_size):
    size = img.shape
    constant = cv.copyMakeBorder(img, 0, need_size - size[0], 0, need_size - size[1], cv2.BORDER_CONSTANT,
                                 value=(0, 0, 0))
    return constant

def get_files(root):
    image_format = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'STM']

    def format_filter(filename):
        for im_format in image_format:
            if filename.lower().endswith(im_format.lower()):
                # lower() 忽略大小格式
                return True
        return False

    img_files = os.listdir(root)
    new_img_files = list(filter(format_filter, img_files))
    file_pathes = [os.path.join(root, filename) for filename in new_img_files]
    return file_pathes

def read_image_save_patchV2(image_path, save_path, model, ps=512):
    try:
        image = gdal.Open(image_path)
        w, h = image.RasterXSize, image.RasterYSize
        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        flag = 1
    except:
        image = imageio.imread(image_path)
        w, h = image.shape[1], image.shape[0]
        flag = 0
    full_image = np.zeros((h, w), dtype=np.uint8)
    stride=int(ps*3//4)
    load_data = LoadTest()
    bounding_range=ps//8

    for y in range(0,h-1,stride):
        for x in range(0,w-1,stride):
            start_y=y
            start_x=x

            end_y=start_y+ps

            if end_y>h:
                start_y=h-ps
                end_y=h

            end_x=start_x+ps

            if end_x>w:
                start_x=w-ps
                end_x=w

            if flag == 1:
                img = image.ReadAsArray(start_x, start_y, ps, ps)
            else:

                img = image[start_y:end_y, start_x:end_x]

            if len(img.shape) == 2:
                im_in_ = img[:, :]
                im_in_ = im_in_[:, :, np.newaxis]
                img = np.concatenate((im_in_, im_in_, im_in_), axis=2)

            img = img.transpose((1, 2, 0))  # c, h, w -> h, w, c
            img = img[:, :, ::-1].copy()  # opencv读取是BGR格式

            tmp_img = load_data(img)
            with torch.no_grad():
                tmp_img = Variable(tmp_img)
                tmp_img = tmp_img.cuda().unsqueeze(0)
                output = model(tmp_img)
            if model.n_classes > 1:
                output = F.softmax(output, dim=1)
            else:
                output = torch.sigmoid(output)

            probs = output.squeeze(0)
            if model.n_classes > 1:
                probs = probs.squeeze(0)

            pred = probs.cpu().detach().numpy()
            pred = np.argmax(pred, axis=0)
            # full_image[start_y:end_y, start_x:end_x] = pred

            if start_x==0 and start_y==0:# 左上
                full_image[start_y:end_y-bounding_range, start_x:end_x-bounding_range] = pred[:-bounding_range,:-bounding_range]

            elif start_x==0 and end_y==h:#左下

                full_image[start_y+bounding_range:end_y, start_x:end_x-bounding_range] = pred[bounding_range:,:-bounding_range]

            elif end_x == w and start_y == 0:#右上
                full_image[start_y:end_y - bounding_range, start_x + bounding_range:end_x] = pred[:-bounding_range,bounding_range:]

            elif end_x==w and end_y==h:#右下
                full_image[start_y+bounding_range:end_y, start_x+bounding_range:end_x] = pred[bounding_range:,bounding_range:]

            elif start_x==0 and start_y!=0 and end_y!=h:#第一列
                full_image[start_y+bounding_range:end_y-bounding_range, start_x:end_x-bounding_range] = pred[bounding_range:-bounding_range,:-bounding_range]

            elif start_x != 0 and start_y == 0 and end_x!=w:#第一行
                full_image[start_y:end_y-bounding_range, start_x+bounding_range:end_x-bounding_range] = pred[:-bounding_range,bounding_range:-bounding_range]


            elif end_x==w and end_y!=h and start_y!=0:#最后一列
                full_image[start_y+bounding_range:end_y-bounding_range, start_x+bounding_range:end_x] = pred[bounding_range:-bounding_range,bounding_range:]

            elif end_x!=w and end_y==h and start_x!=0:#最后一行
                full_image[start_y+bounding_range:end_y, start_x+bounding_range:end_x-bounding_range] = pred[bounding_range:,bounding_range:-bounding_range]

            else:
                full_image[start_y+bounding_range:end_y-bounding_range, start_x+bounding_range:end_x-bounding_range] = pred[bounding_range:-bounding_range,bounding_range:-bounding_range]

    if flag == 1:
        # 创建文件
        driver = gdal.GetDriverByName('GTiff')
        if os.path.basename(image_path)[-3:]!='tif':
            image = driver.Create(save_path + os.path.basename(image_path)[:-4]+'.png', w, h, 1, gdal.GDT_Byte)
        else:
            image = driver.Create(save_path + os.path.basename(image_path), w, h, 1, gdal.GDT_Byte)


        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)
        image.GetRasterBand(1).WriteArray(full_image)
    else:
        if os.path.basename(image_path)[-3:] != 'tif':
             imageio.imwrite(save_path + os.path.basename(image_path)[:-4]+'.png', full_image)
        else:
            imageio.imwrite(save_path + os.path.basename(image_path), full_image)
    del image  # 删除变量,保留数据


class predict_multiclass_model(object):
    def __init__(self, project_class_num, imgpath,  weight_path,save_path, patch_size=512):

        self.project_class_num = project_class_num
        self.imgpath = imgpath
        self.patch_size = patch_size
        self.weight_path = weight_path
        self.save_path = save_path+'/'
        os.makedirs(self.save_path, exist_ok=True)

        self.predict_multiclass_()

    def predict_multiclass_(self):
        '''
        :return:
        '''
        img_pathes = get_files(self.imgpath)

        if not img_pathes:
            print('路径下无tif影像，请查证!')
            return

        if self.project_class_num <= 1:
            self.project_class_num = 2


        model = SLCNet(n_classes=self.project_class_num)

        model.eval()
        if os.path.exists(self.weight_path):
            with torch.no_grad():
                try:
                    model.load_state_dict(torch.load(self.weight_path))
                    print('加载权重成功!')
                except:
                    pretext_model = torch.load(self.weight_path)
                    model_dict = model.state_dict()
                    state_dict = {k: v for k, v in pretext_model.items() if k in model_dict.keys()}

                    model_dict.update(state_dict)
                    model.load_state_dict(model_dict)
                    print('加载部分权重成功!')

        else:
            print('There is no weight file!')
            return
        model.cuda()


        for i, path in enumerate(img_pathes):
            basename = os.path.basename(path)

            print('正在预测:%s, 已完成:(%d/%d)' % (basename, i, len(img_pathes)))
            read_image_save_patchV2(path, self.save_path, model, ps=self.patch_size)

        print('预测完毕!')
