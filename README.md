
# SLCNet

PyTorch implementation of the paper:

> **Long-Range Correlation Supervision for Land-Cover Classification from Remote Sensing Images**

## Environment

The implementation was tested with the following environment:

- Python 3.7
- PyTorch >= 1.5
- torchvision >= 0.6.0
- opencv-python

## Training

### 1. Prepare the Dataset

Prepare your dataset according to the following directory structure.

Assuming that your dataset is stored in `dir_path`:

```text
dir_path/
└── train_data/
    ├── train/
    │   ├── image/
    │   └── label/
    └── val/
        ├── image/
        └── label/
````

where:

* `image/` contains the input remote sensing images.
* `label/` contains the corresponding ground-truth land-cover labels.
* `train/` and `val/` contain the training and validation sets, respectively.

### 2. Configure the Training Script

Create a Python script and add the following code:

```python
from train_model import train_multiclass_model

project_class_num = 6
data_path = 'dir_path'
save_path = 'path for saving the trained model weights'
batchsize = 4
epoch = 60
patch_size = 512  # Size of the image/label tiles

train_multiclass_model(
    project_class_num,
    data_path,
    save_path,
    batchsize,
    epoch,
    patch_size
)
```

Modify the parameters according to your dataset and computing environment:

* `project_class_num`: number of land-cover classes.
* `data_path`: path to the dataset.
* `save_path`: path for saving the trained model weights.
* `batchsize`: training batch size.
* `epoch`: number of training epochs.
* `patch_size`: size of the input image/label tiles.

Run the Python script to start training the SLCNet model.

## Prediction

### 1. Prepare the Test Images

Prepare the test images that will be fed into the trained SLCNet model.

> Only the input images are required for prediction. Ground-truth labels are not needed.

The spatial dimensions of the test images do not need to be fixed to `512 × 512`.

### 2. Configure the Prediction Script

Create a Python script and add the following code:

```python
from predict_model import predict_multiclass_model

project_class_num = 6
imgpath = 'path to the test images'  # Test image sizes can be arbitrary
save_path = 'path for saving the prediction results'
weight_path = 'path to the trained model weights, e.g., *.pkl'
patch_size = 512  # Usually set to 512

predict_multiclass_model(
    project_class_num,
    imgpath,
    weight_path,
    save_path,
    patch_size
)
```

Modify the parameters according to your setup:

* `project_class_num`: number of land-cover classes.
* `imgpath`: path to the test images.
* `save_path`: path for saving the prediction results.
* `weight_path`: path to the trained model weights.
* `patch_size`: patch size used for prediction, typically `512`.

Run the Python script to start prediction with the trained SLCNet model.

## Citation

If you find this project useful for your research, please cite the following paper:

**Yu, D., & Ji, S.**
*Long-Range Correlation Supervision for Land-Cover Classification from Remote Sensing Images.*
IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–14, 2023.


```bibtex
@article{yu2023long,
  author  = {Yu, Dawen and Ji, Shunping},
  title   = {Long-Range Correlation Supervision for Land-Cover Classification from Remote Sensing Images},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {61},
  pages   = {1--14},
  year    = {2023},
  doi     = {10.1109/TGRS.2023.3324706}
}
```

## Update

* **2023.10.13:** Released the SLCNet source code.
* **2023.10.15:** Added the training and testing code for SLCNet.

## Contact

If you have any questions or suggestions, please feel free to contact us:

**Dawen Yu**
Email: [yudawen@whu.edu.cn](mailto:yudawen@whu.edu.cn)

Any questions or discussions are welcome!


