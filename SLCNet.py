import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
resnet = torchvision.models.resnet.resnet50(pretrained=True)

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class ResdiualBlock(nn.Module):
    """
    实现子module：Residual Block
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResdiualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ARFE(nn.Module):

    def __init__(self, out_channels=64):
        super(ARFE, self).__init__()

        in_channels=3
        self.out_channels=out_channels

        self.conv_layers =nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, padding=1, kernel_size=3, stride=1),

            ResdiualBlock(inchannel=out_channels, outchannel=out_channels),
            ResdiualBlock(inchannel=out_channels, outchannel=out_channels),
            ResdiualBlock(inchannel=out_channels, outchannel=out_channels)
            )

        self.gap=nn.AdaptiveAvgPool2d(1)

        self.conv1x1=nn.Sequential(
            nn.Conv2d(out_channels, out_channels, padding=0, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        self.conv_SRF=nn.Sequential(
            nn.Conv2d(out_channels, out_channels, padding=1, kernel_size=3, stride=1,dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv_LRF=nn.Sequential(
            nn.Conv2d(out_channels, out_channels, padding=3, kernel_size=3, stride=1,dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_out=nn.Sequential(
            nn.Conv2d(out_channels, out_channels, padding=0, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,downsampled_image):
        # print(downsampled_image.size())

        x=self.conv_layers(downsampled_image)
        # x=self.down_blocks(x)

        x_s=self.conv_SRF(x)
        x_l=self.conv_LRF(x)

        x_gap=self.gap(x)
        switch=self.conv1x1(x_gap)

        x_s=x_s*switch
        x_l=x_l*(1-switch)
        x=x_s+x_l
        x=self.conv_out(x)
        return x

class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)

class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.out = nn.Linear(in_features=num_units, out_features=num_units, bias=False)
        # self.activate=nn.ReLU(inplace=True)

    def forward(self, query, key, mask=None):
        # print(query.size())
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        ## mask
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)

        ## out = score * V
        # print('values',values.size())
        # print('scores',scores.size())

        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        # print('out',out.size())
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        # print('out',out.size())

        out=self.out(out)
        # print('out',out.size())

        return out, scores

class LRCS(nn.Module):
    def __init__(self, in_channels):
        super(LRCS, self).__init__()
        out_channels=in_channels
        self.msa_v = MultiHeadAttention(in_channels, in_channels, out_channels, 1)
        # self.msa_v2 = MultiHeadAttention(in_channels, in_channels, out_channels, 8)

        self.conv_v = ConvBlock(in_channels=in_channels, out_channels=in_channels, padding=1,kernel_size=3, stride=1)

        self.msa_h = MultiHeadAttention(in_channels, in_channels, out_channels, 1)

        # self.msa_h2 = MultiHeadAttention(in_channels, in_channels, out_channels, 8)
        self.conv_h = ConvBlock(in_channels=in_channels, out_channels=in_channels, padding=1,kernel_size=3, stride=1)

        self.loss_func = nn.L1Loss(reduction='mean')

    def forward(self, x,label=None):

        b,c,h,w=x.size(0),x.size(1),x.size(2),x.size(3)
        if label!=None:
            # label[torch.where(label >= 128)] = label[torch.where(label >= 128)] - 128

            label = label.unsqueeze(1).cuda()
            label = F.interpolate(label.float(), size=(h, w), mode='nearest').long().cuda()

        loss=torch.tensor(0.0).cuda()
        vf = x.clone()#[:, :, :, :].contiguous()  # b,c,h,w
        vf_view = vf.permute(0, 3, 2, 1).reshape(b * w, h, c).contiguous()  # b,w,h,c
        x,vscores = self.msa_v(vf_view, vf_view)#
        x=x.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous() #+ vf
        if label!=None:
            # b 1 h w ->b w,h,1->b w h h
            v_lable= label.permute(0, 3, 2, 1).repeat(1, 1, 1,h)

            v_label_=label.permute(0, 3, 1, 2).repeat(1, 1, h,1)

            v_lable = F.softmax((v_lable==v_label_).float(), dim=3)

            loss+=self.loss_func(v_lable,vscores.reshape(b,w,h,h))

        x=x+vf
        x = self.conv_v(x)

        hf = x.clone()#[:, :, :, :].contiguous()  # b,c,h,w
        hf_view = hf.permute(0, 2, 3, 1).reshape(b * h, w, c).contiguous() # b,h,w,c
        x,hscores = self.msa_h(hf_view, hf_view)
        x=x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous() #+ hf
        x = x + hf
        x = self.conv_h(x)
        if label!=None:
            # b 1 h w ->b h,w,1->b h w w
            h_lable= label.permute(0, 2, 3, 1).repeat(1, 1, 1,w)
            
            h_label_=label.permute(0, 2, 1, 3).repeat(1, 1, w,1)


            h_lable = F.softmax((h_lable==h_label_).float(), dim=3)

            loss+=self.loss_func(h_lable,hscores.reshape(b,h,w,w))

        return x,loss*10


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        if down_x!=None:
            x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

class Side_output(nn.Module):

    def __init__(self, in_channels,out_channels=6):
        super(Side_output, self).__init__()
        self.out_channels=out_channels


        self.segmentation_head =nn.Sequential(
            ConvBlock(in_channels= in_channels, out_channels=64, padding=1, kernel_size=3,stride=1),
            nn.Conv2d(in_channels=64, out_channels=out_channels,padding=0,  kernel_size=1, stride=1)
            )

    def forward(self,x,label=None):

        x = self.segmentation_head(x)
        return x


class SLCNet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes=n_classes
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]

        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bridge = Bridge(2048, 2048)

        self.use_ARFE=True

        cat_channel_number=64 if self.use_ARFE else 0

        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=2048 + cat_channel_number, out_channels=1024,
                                                    up_conv_in_channels=2048, up_conv_out_channels=1024))

        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=1024 + cat_channel_number, out_channels=512,
                                                    up_conv_in_channels=1024, up_conv_out_channels=512))

        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=512 + cat_channel_number, out_channels=256,
                                                    up_conv_in_channels=512, up_conv_out_channels=256))

        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64+ cat_channel_number, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))

        self.up_blocks_final=UpBlockForUNetWithResNet50(in_channels=64, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64)

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)


        ARFEs=[]
        ARFEs.append(ARFE())
        ARFEs.append(ARFE())
        ARFEs.append(ARFE())
        ARFEs.append(ARFE())
        self.ARFEs = nn.ModuleList(ARFEs)

        side_outputs=[]

        side_outputs.append(Side_output(1024,n_classes))
        side_outputs.append(Side_output(512,n_classes))
        side_outputs.append(Side_output(256,n_classes))
        side_outputs.append(Side_output(128,n_classes))

        self.Side_outputs = nn.ModuleList(side_outputs)

        LR = []

        #P1
        LR.append(LRCS(64))
        LR.append(LRCS(256))
        LR.append(LRCS(512))
        LR.append(LRCS(1024))


        self.LR = nn.ModuleList(LR)
        from LovaszSoftmaxloss import lovasz_softmax

        self.loss_func = lovasz_softmax#().cuda()#FocalLoss().cuda()
        self.loss_func2 = torch.nn.CrossEntropyLoss().cuda()#FocalLoss().cuda()



    def forward(self, x,label=None):

        b,c,h,w=x.size(0),x.size(1),x.size(2),x.size(3)
        loss=torch.tensor(0.0).cuda()

        pre_pools = dict()
        pre_pools[f"layer_0"] = x

        x = self.input_block(x)

        #ARFE
        img = F.interpolate(pre_pools[f"layer_0"].float(), size=(x.size(2), x.size(3)), mode='nearest').cuda()
        x2 = self.ARFEs[0](img)

        x1,loss_t=self.LR[0](x,label=label)
        loss += loss_t

        x1=torch.cat((x1,x2),dim=1)
        pre_pools[f"layer_1"] = x1

        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)

            if i == (SLCNet.DEPTH - 1):
                continue

            img=F.interpolate(pre_pools[f"layer_0"].float(), size=(x.size(2), x.size(3)), mode='nearest').cuda()
            x2=self.ARFEs[i-1](img)
            if i>=2:
                x1,loss_t=self.LR[i-1](x,label=label)
                loss+=loss_t
            else:
                x1=x
            x1=torch.cat((x1,x2),dim=1)

            pre_pools[f"layer_{i}"] = x1

        x = self.bridge(x)

        pre_pools['layer_0']=None

        side_outpus=[]
        for i, block in enumerate(self.up_blocks, 1):
            if i <5:
                key = f"layer_{SLCNet.DEPTH - 1 - i}"
                x = block(x, pre_pools[key])

                side_outpus.append(self.Side_outputs[i-1](x))

        x=self.up_blocks_final(x,None)

        x = self.out(x)

        for i in range(4):
            if label!=None:
                side_outpus[i] = F.interpolate(side_outpus[i], size=(h, w), mode='nearest').cuda()
                loss+=0.05*self.loss_func2(side_outpus[i],label.cuda())

        del pre_pools

        if label!=None:

            loss +=self.loss_func(x, label.cuda())

            return x,loss
        else:
            return x

