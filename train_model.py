from torch import nn
import torch
import warnings
import os
from torch.autograd import Variable
import time

from tools import loaderDataset, LoadTrainDataMCB, LoadValDataMCB

from SLCNet import SLCNet

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
warnings.filterwarnings('ignore')


class train_multiclass_model(object):

    def __init__(self, project_class_num,data_path,save_path, batchsize=2, epoch=100,patch_size=512):
        self.patch_size=patch_size

        self.project_class_num=project_class_num
        self.data_path = data_path
        self.save_path = save_path
        os.makedirs(self.save_path,exist_ok=True)

        self.batchsize = batchsize
        self.epoch = epoch
        self.patch_size = patch_size
        self.train()

    def train_multiclass_model_(self, epoch, model, batch_size=3,n_classes=2):
        '''load data'''
        trainData = LoadTrainDataMCB(self.data_path + '/train_data'+'/train/')
        valData = LoadValDataMCB(self.data_path + '/train_data'+'/val/')
        trainLoader = loaderDataset(trainData, batch_size=batch_size, shuffle=True)
        valLoader = loaderDataset(valData, batch_size=batch_size, shuffle=True)

        num_data = len(trainLoader)
        num_flag = num_data // 100 if num_data // 100 == 0 else num_data // 100 + 1
        if not num_flag:
            num_flag = 1
        learn_rate=1e-4

        '''load model and use optimizer'''
        old_weight = self.save_path + '/WEIGHT_last.pkl'
        print('old_weight:',old_weight)
        if os.path.exists(old_weight):
            try:
                model.load_state_dict(torch.load(old_weight))
                print('加载权重成功!')
            except:
                pretext_model = torch.load(old_weight)
                model_dict = model.state_dict()
                state_dict = {k: v for k, v in pretext_model.items() if k in model_dict.keys()}

                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
                print('加载部分权重成功!')

        if torch.cuda.is_available():
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

        if n_classes > 1:
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
        loss_func.cuda()


        st_val_loss, st_train_loss = 10.0, 10.0
        st_val_acc = 0.
        start_time = time.time()

        for ep in range(1, epoch + 1):
            print('doing epoch：{}'.format(ep))

            perLoss, perAcc = 0., 0.

            model.train()
            for idx, (img, label) in enumerate(trainLoader):

                img = Variable(img)
                img = img.cuda()
                img=img.float()
                label=label.float() if n_classes == 1 else label.long()

                output,loss= model(img,label)
                label = Variable(label)
                label = label.cuda()

                perLoss += loss.data.cpu().numpy()

                full_mask = output
                full_mask = torch.argmax(full_mask, dim=1)
                full_mask=full_mask.float() if n_classes == 1 else full_mask.long()

                acc = (full_mask == label).sum().data.cpu().numpy() / (self.patch_size * self.patch_size * self.batchsize)
                perAcc += acc
                if idx % num_flag == 0:
                    print('Train epoch: {} [{}/{} ({:.2f}%)]\tLoss:{:.6f}\tAcc:{:.6f}'.format(
                        ep, idx + 1, len(trainLoader), 100.0 * (idx + 1) / len(trainLoader),
                        loss.data.cpu().numpy(), acc))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t_los_mean = perLoss / len(trainLoader)

            print('Train Epoch: {}, Loss: {:.4f}'.format(ep, t_los_mean))

            ''' val '''
            model.eval()
            perValLoss = 0.
            perValAcc = 0.
            print('正在进行验证模型，请稍等...')
            for idx, (img, label) in enumerate(valLoader):

                with torch.no_grad():
                    img = Variable(img)
                    img = img.cuda()
                    img = img.float()
                    label = label.float() if n_classes == 1 else label.long()

                    output,loss= model(img,label)

                    label = Variable(label)
                    label = label.cuda()


                perValLoss += loss.data.cpu().numpy()
                full_mask = output
                full_mask = torch.argmax(full_mask, dim=1)
                full_mask=full_mask.float() if n_classes == 1 else full_mask.long()

                valacc = (full_mask == label).sum().data.cpu().numpy() / (self.patch_size* self.patch_size * self.batchsize)
                perValAcc += valacc

            val_los_mean = perValLoss / len(valLoader)
            val_acc_mean = perValAcc / len(valLoader)

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(val_los_mean, val_acc_mean))
            # if st_val_acc < val_acc_mean:
            #
            #     st_val_acc = val_acc_mean
            #     # 仅保存和加载模型参数
            #     print(r'进行权重保存-->>\nEpoch：{}\t\nTrainLoss:{:.4f}'
            #           ''.format(ep, float(t_los_mean)))

                # torch.save(model.state_dict(), self.save_path + r'/WEIGHT_best.pkl')

            duration1 = time.time() - start_time
            start_time = time.time()
            print('train running time: %.2f(minutes)' % (duration1 / 60))
            if ep in [epoch//3, epoch//3*2]:

                learn_rate = learn_rate*0.1
                print('Drop learning rate to', learn_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learn_rate

            torch.save(model.state_dict(), self.save_path + '/WEIGHT_last.pkl')

    def train(self):
        if self.project_class_num<=1:
            self.project_class_num=2

        model = SLCNet(n_classes=self.project_class_num)
        self.train_multiclass_model_(epoch=self.epoch, model=model, batch_size=self.batchsize,n_classes=self.project_class_num)


if __name__ == '__main__':
    print('training part!')