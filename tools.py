import os
from tqdm import tqdm
import cv2
from shutil import move
import torch
import os
import cv2 as cv
import numpy as np
from torchvision.transforms import Normalize, ToTensor, Compose
import random
from torch.utils import data
from skimage import exposure
import shutil

def tianchong_xy(img,need_size):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, need_size - size[0], 0, need_size - size[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return constant

def tianchong_x(img,need_size):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, 0, 0, need_size - size[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return constant

def tianchong_y(img,need_size):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, need_size - size[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return constant

def tianchong_xy_(img,need_size):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, need_size - size[0], 0, need_size - size[1], cv2.BORDER_CONSTANT, value=(0))
    return constant

def tianchong_x_(img,need_size):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, 0, 0, need_size - size[1], cv2.BORDER_CONSTANT, value=(0))
    return constant

def tianchong_y_(img,need_size):
    size = img.shape
    constant = cv2.copyMakeBorder(img, 0, need_size - size[0], 0, 0, cv2.BORDER_CONSTANT, value=(0))
    constant = cv2.copyMakeBorder(img, 0, need_size - size[0], 0, 0, cv2.BORDER_CONSTANT, value=(0))
    return constant

def getfilename_multi(path,str):
    f_list = os.listdir(path)
    filename=[]
    for i in f_list:
      for j in range(len(str)):
        if os.path.splitext(i)[1] == str[j]:
          filename.append(i)
    return filename

def clip_stride_images(image, image_path, save_path, patch_size=512, overlap_rate=0.25, num=0):

    stride = patch_size-int(patch_size*overlap_rate)  # 移动步伐
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    basename = os.path.basename(image_path).split('.')[0]

    m, n, _ = image.shape

    # 根据滑动步伐和原始影像大小定义整倍数的影像大小
    mtimes = (m-patch_size) // stride if (m - patch_size) % stride == 0 else (m - patch_size)//stride + 1
    ntimes = (n-patch_size) // stride if (n - patch_size) % stride == 0 else (n - patch_size)//stride + 1

    tmp_num = 0
    for r in range(mtimes + 1):
        for c in range(ntimes + 1):
            if r == mtimes and c != ntimes:
                tmp_img = image[-patch_size:, c*stride:c*stride + patch_size, :]
            elif c == ntimes and r != mtimes:
                tmp_img = image[r*stride:r*stride + patch_size, -patch_size:, :]
            elif r == mtimes and c == ntimes:
                tmp_img = image[-patch_size:, -patch_size:, :]
            else:
                tmp_img = image[r*stride:r*stride + patch_size, c*stride:c*stride + patch_size, :]
            cv2.imwrite(save_path+'/'+basename + '_' + str(tmp_num) + '.tif', tmp_img)
            tmp_num += 1

    return

def clip_stride_labels(label, label_path, save_path, patch_size=512, overlap_rate=0.25):

    stride = patch_size-int(patch_size*overlap_rate)  # 移动步伐
    # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    basename = os.path.basename(label_path).split('.')[0]
    m, n = label.shape

    # 根据滑动步伐和原始影像大小定义整倍数的影像大小
    mtimes = (m-patch_size) // stride if (m - patch_size) % stride == 0 else (m - patch_size)//stride + 1
    ntimes = (n-patch_size) // stride if (n - patch_size) % stride == 0 else (n - patch_size)//stride + 1


    tmp_num = 0
    for r in range(mtimes + 1):
        for c in range(ntimes + 1):
            if r == mtimes and c != ntimes:
                tmp_label = label[-patch_size:, c*stride:c*stride + patch_size]
            elif c == ntimes and r != mtimes:
                tmp_label = label[r*stride:r*stride + patch_size, -patch_size:]
            elif r == mtimes and c == ntimes:
                tmp_label = label[-patch_size:, -patch_size:]
            else:
                tmp_label = label[r*stride:r*stride + patch_size, c*stride:c*stride + patch_size]
            cv2.imwrite(save_path+'/'+basename + '_' + str(tmp_num) + '.tif', tmp_label)
            tmp_num += 1
    return

def generate_val_data(ori_image_path, ori_label_path, target_image,target_label, move_rate=0.1):

    image_names = getfilename_multi(ori_image_path,['.tif'])
    all_num = len(image_names)

    val_num = int(all_num * move_rate)
    train_num=all_num-val_num

    np.random.shuffle(image_names)

    for i in range(val_num):
        cur_ori_image_path=ori_image_path+'/'+image_names[i]
        cur_ori_label_path=ori_label_path+'/'+image_names[i]

        cur_target_image=target_image+'/'+image_names[i]
        cur_target_label=target_label+'/'+image_names[i]

        move(cur_ori_image_path, cur_target_image)
        move(cur_ori_label_path, cur_target_label)
    return train_num, val_num

def make_multiclass_train_data(input_image_path, input_label_path, project_main_path, overlap_rate=1/8,patch_size=512):

    '''train'''
    save_image_path = project_main_path + '/train_data/train/image'
    save_label_path = project_main_path + '/train_data/train/label'
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)

    '''val'''
    save_val_image_path = project_main_path + '/train_data/val/image'
    save_val_label_path = project_main_path + '/train_data/val/label'
    os.makedirs(save_val_image_path, exist_ok=True)
    os.makedirs(save_val_label_path, exist_ok=True)

    image_names = getfilename_multi(input_image_path,['.tif','.png','.jpg','.bmp'])
    if len(image_names)==0:
        print('路径下没有影像!')
        return

    # patch_size = 512
    for image_name in tqdm(image_names):

        cur_image_path=input_image_path+'/'+image_name
        cur_label_path=input_label_path+'/'+image_name[:-3]+'png'

        if os.path.exists(cur_label_path)==False:

            continue

        cur_image=cv2.imread(cur_image_path, cv2.IMREAD_COLOR)
        cur_label=cv2.imread(cur_label_path, cv2.IMREAD_GRAYSCALE)

        if cur_image.shape[:2] != cur_label.shape[:2]:
            continue

        if cur_image.shape[0]<patch_size:
            cur_image=tianchong_y(cur_image,patch_size)
            cur_label=tianchong_y_(cur_label,patch_size)

        if cur_image.shape[1]<patch_size:
            cur_image=tianchong_x(cur_image,patch_size)
            cur_label=tianchong_x_(cur_label,patch_size)

        if cur_image.shape[0] < patch_size and cur_image.shape[1]:
            cur_image = tianchong_xy(cur_image, patch_size)
            cur_label = tianchong_xy_(cur_label, patch_size)

        clip_stride_images(cur_image,cur_image_path, save_image_path, overlap_rate=overlap_rate, patch_size=patch_size)
        clip_stride_labels(cur_label,cur_label_path, save_label_path, overlap_rate=overlap_rate, patch_size=patch_size)


    '''选出验证集'''
    print('将训练集中的1/10移出作为验证集!')
    train_num, val_num = generate_val_data(save_image_path, save_label_path, save_val_image_path, save_val_label_path, move_rate=0.1)
    print('数据准备完毕, 训练集/验证集: %d/%d' % (train_num, val_num))
    return

def read_txt(txt_path):
    f = open(txt_path)
    line = f.readline()
    data_list = []
    while line:
        num = list(map(str,line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    data_array = np.array(data_list)
    return data_array

def geocoor_to_imgcoor(xml_path,polygon):

    tfw = read_txt(xml_path)
    x_step = float(tfw[0][0])
    y_step = float(tfw[3][0])
    x_start = float(tfw[4][0])
    y_start = float(tfw[5][0])

    new_polygon=[]
    for point in polygon:
        geo_x,geo_y=point[0],point[1]
        img_x=int((geo_x-x_start)/x_step)
        img_y=int((geo_y-y_start)/y_step)
        new_polygon.append([img_x,img_y])

    return new_polygon


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

class Balance(object):
    def __init__(self, n):
        # 1 线性拉伸; 255 直方图均衡化
        # 把原始图像的灰度直方图从比较集中的某个灰度区间变成在全部灰度范围内的均匀分布。
        self.n = n

    def __call__(self, img, *args, **kwargs):
        if self.n == 255:
            split = cv.split(img)
            for i in range(3):
                cv.equalizeHist(split[i], split[i])
            img = cv.merge(split)

        elif self.n == 1:
            # print("执行线性拉伸")
            split = cv.split(img)
            for i in range(3):
                split[i] = exposure.rescale_intensity(split[i])
            img = cv.merge(split)
        return img

class GaussioanBlurSize(object):
    def __init__(self, size,sigma=None):
        # 随机size:(0, 1)，进行高斯平滑
        self.KSIZE = size * 2 + 3
        self.sigma = sigma

    def __call__(self, img):
        n = random.randint(0, 12) if self.sigma==None else 1 # 1/4进行高斯滤波
        if n == 0:
            sigma = 2.2
        elif n == 1:
            sigma = 1.5
        elif n == 2:
            sigma = 3
        else:
            return img
        dst = cv.GaussianBlur(img, (self.KSIZE, self.KSIZE), sigma, self.KSIZE)
        return dst

class ToLabelB(object):

    def __call__(self, label):
        label[label >= 127] = 1
        label[label < 0] = 0
        return torch.from_numpy(label).float()

class LoadTrainDataMCB(data.Dataset):
    def __init__(self, root):
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))


        self.balance = Balance(np.random.randint(0, 4))  # 1/4的概率进行线性拉伸
        self.gaussBS = GaussioanBlurSize(np.random.randint(0, 2))  # 1/4的概率进行高斯滤波

        self.img_transform = ToTensor()
        self.label_transform = ToLabelB()

        self.img_paths = get_files(self.root + 'image')
        self.label_paths = get_files(self.root + 'label')

        self.image_num=len(self.img_paths)

    def __getitem__(self, index):

        img = cv.imread(self.img_paths[index], cv.IMREAD_COLOR)
        label = cv.imread(self.label_paths[index], cv.IMREAD_GRAYSCALE)
        size, _ = label.shape

        id=random.random()
        if id< 0.25:
            img = cv.flip(img, 1)  # 水平镜像
            label = cv.flip(label, 1)
        elif id<0.5:
            img = cv.flip(img, 0)  # 垂直镜像
            label = cv.flip(label, 0)

        img = self.gaussBS(img)
        img = self.balance(img)

        img = self.img_transform(img)
        label = self.label_transform(label)

        return img, label

    def __len__(self):
        return len(self.img_paths)

class LoadValDataMCB(data.Dataset):
    def __init__(self, root):
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))


        self.img_transform = ToTensor()
        self.label_transform = ToLabelB()

        self.img_paths = get_files(self.root + 'image')
        self.label_paths = os.path.join(self.root, "label\\")

    def __getitem__(self, index):
        basename = os.path.basename(self.img_paths[index])

        img = cv.imread(self.img_paths[index], cv.IMREAD_COLOR)
        label = cv.imread(self.label_paths + basename, cv.IMREAD_GRAYSCALE)
        size, _ = label.shape

        img = self.img_transform(img)
        label = self.label_transform(label)
        return img, label

    def __len__(self):
        return len(self.img_paths)

def loaderDataset(dataset, batch_size, shuffle=True):
    return data.DataLoader(dataset, batch_size, shuffle=shuffle,drop_last=True)


class LoadTest(object):
    def __init__(self):
        self.img_transform = ToTensor()

    def __call__(self, img):
        img = self.img_transform(img)
        return img







