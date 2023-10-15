# SLCNet
Pytorch code for the paper "Long-Range Correlation Supervision for Land-Cover Classification from Remote Sensing Images".

## Environment
1. Python 3.7    
2. pytoch>=1.5 torchvision>=0.6.0    
3. opecncv-python

## Training
1.prepare your data    
    
    
    The organization of your data is as follows:
    assuming your data is in the 'dir_path',
    'dir_path' has a subfolder, i.e.'train_data'
    'train_data' has two subfolders, i.e.'train' and 'val'
    'train'and 'val' both have two subfolders, i.e., 'image' and 'label'

2.create a python file  and input:

   
    from train_model import train_multiclass_model

    project_class_num=6
    data_path='dir_path'
    save_path='path for saving the training model weight'
    batchsize=4
    epoch=60
    patch_size=512 #the size of your image/label tiles  
    train_multiclass_model(project_class_num,data_path,save_path, batchsize, epoch,patch_size)
    
    #run the python file and the SLCNet model start training
    
## Predicting
1. prepare your data, just the images would be feded into the model 

2. create a python file and input:

   
    from predict_model import predict_multiclass_model

    project_class_num=6
    imgpath='test images path'#the sizes of the testing images can be random.
    save_path='path for saving the predicting results'
    weight_path='model weight path, .../*.pkl'
    patch_size=512 #usually 512
    predict_multiclass_model(project_class_num, imgpath,  weight_path,save_path, patch_size)
    
    #run the python file and the SLCNet model start predicting


### Citation [link](https://doi.org/10.1109/TGRS.2023.3324706)

If you find this project useful for your research, please cite this work.
    
    Yu, D., & Ji, S. "Long-Range Correlation Supervision for Land-Cover Classification from Remote Sensing Images",IEEE Transactions on Geoscience and Remote Sensing (TGRS), vol. 61, pp. 1-14, 2023.

### Update
1. 2023.10.13, We released the code of SCLNet. 
2. 2023.10.15, We added the code for training and testing the SCLNet.

Contact: yudawen@whu.edu.cn. Any questions or discussions are welcomed!

