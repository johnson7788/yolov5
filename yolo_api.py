#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/20 2:20 下午
# @File  : yolo_api.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 推理和训练的api
######################################################
# 包括训练接口api和预测接口api
# /api/train
# /api/predict
######################################################

import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")

import os
import time
from pathlib import Path
import requests
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


from flask import Flask, request, jsonify, abort

app = Flask(__name__)


class YOLOModel(object):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.label_list = ['table', 'figure', 'equation']
        #给每个类别的候选框设置一个颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.label_list]
        self.num_labels = len(self.label_list)
        # 判断使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # 预测的batch_size大小
        self.train_batch_size = 8
        # 预测的batch_size大小
        self.predict_batch_size = 16
        #模型的名称或路径
        self.weights = 'runs/train/exp/weights/last.pt'      # 'yolov5s.pt'
        self.source = 'images_dir'  #图片目录
        self.img_size = 640   #像素
        self.conf_thres = 0.25  #置信度
        self.iou_thres = 0.45  #IOU的NMS阈值
        self.view_img = False   #是否显示图片的结果
        self.save_img = True    #保存图片预测结果
        self.save_conf = False  #同时保存置信度到save_txt文本中
        self.classes = None  # 0, 1, 2 ，只过滤出我们希望的类别, None表示保留所有类别
        self.agnostic_nms = False #使用nms算法
        self.project = 'runs/api' #项目保存的路径
        self.image_dir = os.path.join(self.project, 'images')   #保存从网络下载的图片
        self.predict_dir = os.path.join(self.project, 'predict')   #保存预测结果
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
        self.load_predict_model()

    def load_train_model(self):
        """
        初始化训练的模型
        :return:
        """
        pass
        logger.info(f"训练模型{self.tuned_checkpoint_S}加载完成")

    def load_predict_model(self):
        # Load model
        self.predict_model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.predict_model.stride.max())  # model stride
        logger.info(f"预测模型{self.weights}加载完成")

    def download_file(sefl, url, save_dir):
        """
        我们返回绝对路径
        :param url: eg: http://127.0.0.1:9090/2007.158710001-01.jpg
        :param save_dir: eg: /tmp/
        :return:  /tmp/2007.158710001-01.jpg
        """
        local_filename = url.split('/')[-1]
        save_dir_abs = Path(save_dir).absolute()
        save_file = os.path.join(save_dir_abs, local_filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return save_file

    def detect(self, data):
        """
        :param data: 图片数据的列表 [image1, image2]
        :return:
        """
        #检查设置的图片的大小和模型的步长是否能整除
        imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
        #下载数据集, images保存图片的本地的路径
        images = []
        for url in data:
            image = self.download_file(url, self.image_dir)
            images.append(image)
        #设置数据集
        dataset = LoadImages(path=self.image_dir, img_size=imgsz, stride=self.stride)
        # 这里我们重设下images，我们只要自己需要的images既可
        dataset.files = images
        # 设置模型
        predict_model = self.predict_model
        # 运行推理
        if self.device.type != 'cpu':
            predict_model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(predict_model.parameters())))  # run once
        #计算耗时
        t0 = time.time()
        # path是图片的路径，img是图片的改变size后的numpy格式[channel, new_height,new_witdh], im0s是原始的图片,[height, width,channel], eg: (2200, 1700, 3), vid_cap 是None如果是图片，只对视频有作用
        for path, img, im0s, vid_cap in dataset:
            # 如果是GPU，会放到GPU上
            img = torch.from_numpy(img).to(self.device)
            #转换成float
            img = img.float()
            #归一化
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #扩展一个batch_size维度, [batch_isze, channel, new_height, new_witdh], eg:  torch.Size([1, 3, 640, 512])
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            #开始推理, time_synchronized是GPU的同步
            t1 = time_synchronized()
            # pred模型的预测结果 [batch_size,x,x ] eg: torch.Size([1, 20160, 8]), 8代表 (x1, y1, x2, y2, conf, cls1, cls2, cls3...), 前4个是bbox坐标，conf是置信度，cls是类别的，cls1代表是类别1的概率
            pred = predict_model(img, augment=False)[0]

            #使用 NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # s 是初始化一个空字符串，用于打印预测结果，im0是原始图片, frame是对于视频而言的
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                #p原始图片的绝对路径
                p = Path(p)  # to Path
                save_path = os.path.join(self.predict_dir, p.name)  #预测后的保存的图片的路径
                s += '图片尺寸%gx%g, ' % img.shape[2:]  # print string, eg '640x480 '
                # 图片的width,height, width, height, eg: tensor([1700, 2200, 1700, 2200]), 用于下面的归一化
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # bbox 放大到原始图像的大小，从img_size 到原始图像 im0 尺寸， bbox左上角的x1，y1, 右下角的x2,y2
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    #最后一个维度的最后一位是预测的结果
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n}个{self.label_list[int(c)]}{'s' * (n > 1)} bbox, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_img or self.view_img:  # Add bbox to image
                            label = f'{self.label_list[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)
                    print(f'{s}完成. 耗时({t2 - t1:.3f}s)')
                else:
                    print(f'{s}完成. 没有发现目标,耗时({t2 - t1:.3f}s)')

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    print(f"保存结果到 {self.predict_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

    def do_train(self, data):
        """
        训练模型, 数据集分成2部分，训练集和验证集, 默认比例9:1
        :param data: 输入的数据，注意如果做truncated，那么输入的数据为 []
        :return:
        """
        pass
        logger.info(f"训练完成")
        return "Done"

@app.route("/api/predict", methods=['POST'])
def predict():
    """
    接收POST请求，获取data参数
    Args:
        test_data: 需要预测的数据，是一个图片的url列表, [images1, images2]
    Returns: 返回格式是 [(predicted_label, predict_score),...]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = model.detect(test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)


@app.route("/api/train", methods=['POST'])
def train():
    """
    接收data参数，
    Args:
        data: 训练的数据，是一个图片列表, [images1, images2,...]
    Returns:
    """
    jsonres = request.get_json()
    data = jsonres.get('data', None)
    logger.info(f"收到的数据是:{data}, 进行训练")
    results = model.do_train(data)
    return jsonify(results)





if __name__ == "__main__":
    model = YOLOModel()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
