<a align="left" href="https://apps.apple.com/app/id1452689527" target="_blank">
<img width="800" src="https://user-images.githubusercontent.com/26833433/98699617-a1595a00-2377-11eb-8145-fc674eb9b1a7.jpg"></a>
&nbsp

<a href="https://github.com/ultralytics/yolov5/actions"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>

该repository代表Ultralytics对未来的对象检测方法的开源研究，并结合了在匿名客户数据集上数千小时的训练和评估过拟合中汲取的经验教训和最佳实践。 **所有代码和模型都在积极开发中，如有更改或删除，恕不另行通知, 使用后果自负。

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure (click to expand)</summary>
  
<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313219-f1d70e00-9af5-11eb-9973-52b1f98d321a.png"></p>
</details>
<details>
  <summary>Figure Notes (click to expand)</summary>
  
  * GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. 
  * EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.
  * **Reproduce** by `python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`
</details>

- **April 11, 2021**: [v5.0 release](https://github.com/ultralytics/yolov5/releases/tag/v5.0): YOLOv5-P6 1280 models, [AWS](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart), [Supervise.ly](https://github.com/ultralytics/yolov5/issues/2518) and [YouTube](https://github.com/ultralytics/yolov5/pull/2752) integrations.
- **January 5, 2021**: [v4.0 release](https://github.com/ultralytics/yolov5/releases/tag/v4.0): nn.SiLU() activations, [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme) logging, [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/) integration.
- **August 13, 2020**: [v3.0 release](https://github.com/ultralytics/yolov5/releases/tag/v3.0): nn.Hardswish() activations, data autodownload, native AMP.
- **July 23, 2020**: [v2.0 release](https://github.com/ultralytics/yolov5/releases/tag/v2.0): improved model definition, training and mAP.


## 预训练模型

<<<<<<< HEAD
| Model | size | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>V100</sub> | FPS<sub>V100</sub> || params | GFLOPS |
|---------- |------ |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases)    |640 |36.8     |36.8     |55.6     |**2.2ms** |**455** ||7.3M   |17.0
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases)    |640 |44.5     |44.5     |63.1     |2.9ms     |345     ||21.4M  |51.3
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases)    |640 |48.1     |48.1     |66.4     |3.8ms     |264     ||47.0M  |115.4
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases)    |640 |**50.1** |**50.1** |**68.7** |6.0ms     |167     ||87.7M  |218.8
| | | | | | | || |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases) + TTA |832 |**51.9** |**51.9** |**69.6** |24.9ms |40      ||87.7M  |1005.3

<!--- 
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |640 |49.0     |49.0     |67.4     |4.1ms     |244     ||77.2M  |117.7
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |1280 |53.0     |53.0     |70.8     |12.3ms     |81     ||77.2M  |117.7
--->

** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results denote val2017 accuracy.  
** All AP numbers are for single-model single-scale without ensemble or TTA. **Reproduce mAP** by `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`  
** Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and includes image preprocessing, FP16 inference, postprocessing and NMS. NMS is 1-2ms/img.  **Reproduce speed** by `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`  
** 使用默认设置和超参数将所有checkpoint训练到300个epoch  (no autoaugmentation). 
** Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) runs at 3 image sizes. **Reproduce TTA** by `python test.py --data coco.yaml --img 832 --iou 0.65 --augment` 
=======
[assets]: https://github.com/ultralytics/yolov5/releases

Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPS<br><sup>640 (B)
---   |---  |---        |---         |---             |---                |---|---              |---
[YOLOv5s][assets]    |640  |36.7     |36.7     |55.4     |**2.0** | |7.3   |17.0
[YOLOv5m][assets]    |640  |44.5     |44.5     |63.1     |2.7     | |21.4  |51.3
[YOLOv5l][assets]    |640  |48.2     |48.2     |66.9     |3.8     | |47.0  |115.4
[YOLOv5x][assets]    |640  |**50.4** |**50.4** |**68.8** |6.1     | |87.7  |218.8
| | | | | | || |
[YOLOv5s6][assets]   |1280 |43.3     |43.3     |61.9     |**4.3** | |12.7  |17.4
[YOLOv5m6][assets]   |1280 |50.5     |50.5     |68.7     |8.4     | |35.9  |52.4
[YOLOv5l6][assets]   |1280 |53.4     |53.4     |71.1     |12.3    | |77.2  |117.7
[YOLOv5x6][assets]   |1280 |**54.4** |**54.4** |**72.0** |22.4    | |141.8 |222.9
| | | | | | || |
[YOLOv5x6][assets] TTA |1280 |**55.0** |**55.0** |**72.0** |70.8 | |-  |-

<details>
  <summary>Table Notes (click to expand)</summary>
  
  * AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results denote val2017 accuracy.  
  * AP values are for single-model single-scale unless otherwise noted. **Reproduce mAP** by `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`  
  * Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and includes FP16 inference, postprocessing and NMS. **Reproduce speed** by `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`  
  * All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
  * Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) includes reflection and scale augmentation. **Reproduce TTA** by `python test.py --data coco.yaml --img 1536 --iou 0.7 --augment`
</details>
>>>>>>> upstream/master


## 依赖

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```


## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️ RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Supervisely Ecosystem](https://github.com/ultralytics/yolov5/issues/2518)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)


## 环境

YOLOv5可以在以下任何经过验证的最新环境中运行 (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab and Kaggle** notebooks with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
- **Amazon** Deep Learning AMI. See [AWS Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart)
- **Docker Image**. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>


<<<<<<< HEAD
## 推理
### 方法1
detect.py在各种源文件上进行推理，并自动从  [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) 下载模型并把推理结果保存到 `runs/detect`.
=======
## Inference

`detect.py` runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
>>>>>>> upstream/master
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```


示例2： 在以下示例图像上进行推理 `data/images`，  自动下载yolov5s.pt模型:
```bash
$ python detect.py --source data/images --weights yolov5s.pt --conf 0.25

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', project='runs/detect', save_conf=False, save_txt=False, source='data/images/', update=False, view_img=False, weights=['yolov5s.pt'])
YOLOv5 v4.0-96-g83dc1b4 torch 1.7.0+cu101 CUDA:0 (Tesla V100-SXM2-16GB, 16160.5MB)

Fusing layers... 
<<<<<<< HEAD
Model Summary: 232 layers, 7459581 parameters, 0 gradients
image 1/2 data/images/bus.jpg: 640x480 4 persons, 1 buss, 1 skateboards, Done. (0.012s)
image 2/2 data/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.012s)
Results saved to runs/detect/exp
Done. (0.113s)

生成的结果如下，bbox已经在图片中框选出来
#ls runs/detect/exp/
bus.jpg    zidane.jpg

=======
Model Summary: 224 layers, 7266973 parameters, 0 gradients, 17.0 GFLOPS
image 1/2 /content/yolov5/data/images/bus.jpg: 640x480 4 persons, 1 bus, Done. (0.010s)
image 2/2 /content/yolov5/data/images/zidane.jpg: 384x640 2 persons, 1 tie, Done. (0.011s)
Results saved to runs/detect/exp2
Done. (0.103s)
>>>>>>> upstream/master
```
<img width="500" src="https://user-images.githubusercontent.com/26833433/97107365-685a8d80-16c7-11eb-8c2e-83aac701d8b9.jpeg">  

### PyTorch Hub
方法2，使用 PyTorch Hub, 对于大量数量，使用批次推理 YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36):
```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images

# Inference
results = model(imgs)
results.print()  # or .show(), .save()
```

## 训练模型
### 从头开始训练
训练模型，运行以下命令以重现结果[COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (数据集自动下载). 训练时间： YOLOv5s/m/l/x are 2/4/6/8 天在单颗GPU上 V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

### 继续训练
训练自定义的数据,使用yolov5s.pt 预训练模型继续训练, 250张图片，3个epoch，耗时约1分钟
```buildoutcfg
python train.py --img 640 --batch 16 --epochs 3 --data pdfmini.yaml --weights yolov5s.pt --nosave --cache
python train.py --batch 16 --epochs 100 --data pdfmini.yaml --weights yolov5s.pt --nosave --cache

训练结果
tree runs/train/exp/
runs/train/exp/
├── confusion_matrix.png
├── events.out.tfevents.1613795812.wacserver3.21644.0
├── F1_curve.png
├── hyp.yaml
├── labels_correlogram.jpg
├── labels.jpg
├── opt.yaml
├── P_curve.png
├── PR_curve.png
├── R_curve.png
├── results.png
├── results.txt
├── test_batch0_labels.jpg
├── test_batch0_pred.jpg
├── test_batch1_labels.jpg
├── test_batch1_pred.jpg
├── test_batch2_labels.jpg
├── test_batch2_pred.jpg
├── train_batch0.jpg
├── train_batch1.jpg
├── train_batch2.jpg
└── weights
    ├── best.pt
    └── last.pt
```
文件结果展示
```buildoutcfg
cat runs/train/exp/results.txt
    epoch/total  gpu_mem（占用的GPU)  box      obj      cls         total        targets   img_size  
     98/99     3.03G                0.02208   0.01128  0.002249   0.03561        45       640    0.9738     0.988    0.9928    0.7999   0.01238  0.005041 0.0005524
     99/99     3.03G                 0.02176   0.01176  0.003855   0.03737        44       640    0.9491    0.9808    0.9848    0.7529    0.0135  0.005859 0.0007229                  

```


## 使用自定义的训练的api
```buildoutcfg
python yolo_api.py
测试
python yolo_api_test.py
[['/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg', [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0]], [0.9033942818641663, 0.8640206456184387], ['图像', '公式']], ['/Users/admin/git/yolov5/runs/api/images/A_Comprehensive_Survey_of_Grammar_Error_Correction0001-21.jpg', [[864.0, 132.0, 1608.0, 459.0], [865.0, 1862.0, 1602.0, 1944.0], [863.0, 1655.0, 1579.0, 1753.0], [115.0, 244.0, 841.0, 327.0], [116.0, 398.0, 837.0, 486.0], [124.0, 130.0, 847.0, 235.0]], [0.9183754920959473, 0.8920623660087585, 0.8884797692298889, 0.8873556852340698, 0.8276346325874329, 0.5401338934898376], ['表格', '公式', '公式', '公式', '公式', '公式']], ['/Users/admin/git/yolov5/runs/api/images/2007.158710001-09.jpg', [], [], []], ['/Users/admin/git/yolov5/runs/api/images/Relation-Aware_Collaborative_Learning_for_Uni%EF%AC%81ed_Aspect-Based_Sentiment_Analysis0001-02.jpg', [[170.0, 165.0, 818.0, 542.0]], [0.9089462757110596], ['表格']]]
```

## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## About Us

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:
- **Cloud-based AI** systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at https://www.ultralytics.com. 


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit https://www.ultralytics.com or email Glenn Jocher at glenn.jocher@ultralytics.com. 



# wiki
## [自定义训练集](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
训练示例colab: https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb
### 安装依赖  Python>=3.8 and PyTorch>=1.7
```buildoutcfg
$ git clone https://github.com/ultralytics/yolov5  # clone repo
$ cd yolov5
$ pip install -r requirements.txt  # install dependencies
```
### 创建数据集
####  1. 创建dataset.yaml
COCO128是一个小型教程数据集，由COCO train2017中的前128张图像组成。 这些相同的128张图像用于训练和验证，以验证我们的训练pipeline是否能够拟合。 如下所示，data/coco128.yaml是数据集配置文件，该文件定义了1）用于自动下载的可选下载命令/URL，2）训练图像目录的路径（或带有*的*.txt文件的路径的训练图像列表），3）与我们的验证图像相同，4）类别数量，5）类别名称列表：
```buildoutcfg
# cat data/coco128.yaml
# download command/URL (optional)
download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../coco128/images/train2017/
val: ../coco128/images/train2017/

# number of classes
nc: 80

# class names
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']
```

#### 2. 创建标签，为图像打标
使用CVAT，makesense.ai或Labelbox之类的工具标注图像后，将label导出为YOLO格式，每个图像一个*.txt文件（如果图像中没有对象，则不需要*.txt文件）。 *.txt文件规范为： 
```buildoutcfg
每个对象一行
每行都是label x_center y_center width height 格式。
框坐标必须采用标准化的xywh格式（0~1）。 如果您的框以像素为单位，则将x_center和width除以图像宽度，将y_center和height除以图像高度。
类别编号为零索引（从0开始）。
```
<img src="https://user-images.githubusercontent.com/26833433/91506361-c7965000-e886-11ea-8291-c72b98c25eec.jpg" width="600">

与上图相对应的标签文件包含2个人（0类）和一条领带（27类）： 
<img src="https://user-images.githubusercontent.com/26833433/98809572-0bc4d580-241e-11eb-844e-eee756f878c2.png" width="600">

#### 3.整理图片和标签的文件夹位置
根据以下样本整理训练和val图像和标签。 在此样本中，我们假设/coco128在/yolov5目录旁边。 YOLOv5通过用/labels/替换images目录中的/images/的最后一个实例，自动为每个图像定位label。 例如： 
```buildoutcfg
#只有目录和后缀不一样
coco/images/train2017/000000109622.jpg  # image
coco/labels/train2017/000000109622.txt  # label
```
<img src="https://user-images.githubusercontent.com/26833433/83666389-bab4d980-a581-11ea-898b-b25471d37b83.jpg" width="600">

#### 4.选择一个预训练模型
选择一个预训练的模型以开始训练。 在这里，我们选择YOLOv5s，这是最小，最快的模型。 有关所有模型的完整比较，请参见我们的自述表。 
<img src="https://user-images.githubusercontent.com/26833433/103595982-ab986000-4eb1-11eb-8c57-4726261b0a88.png" width="600">

### 5.开始训练
通过指定数据集，批次大小，图像大小以及预训练的--weights yolov5s.pt（推荐）或随机初始化的--weights'' --cfg yolov5s.yaml（不推荐），在COCO128上训练YOLOv5s模型。 可从最新的YOLOv5版本中自动下载预训练的权重。 
```buildoutcfg
# Train YOLOv5s on COCO128 for 5 epochs
$ python train.py --img 640 --batch 16 --epochs 5 --data coco128.yaml --weights yolov5s.pt
```
所有训练结果都将以递增的运行目录（即，runs/train/exp2，runs/train/exp3等）保存到runs/train/中。有关更多详细信息，请参见Google Colab Notebook的Training部分。 

### 可视化部分

