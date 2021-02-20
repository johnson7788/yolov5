#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/12/23 5:02 下午
# @File  : main_api_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试yolo_api.py

import requests
import json

def dopredict(test_data, host="127.0.0.1"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}:5000/api/predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    result = r.json()
    print(result)
    return result

def dotrain(train_data, host="127.0.0.1"):
    """
    使用train_data训练模型
    :param train_data:
    :param host:
    :return:
    """
    url = f"http://{host}:5000/api/train"
    data = {'data': train_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    result = r.json()
    print(result)
    return result

if __name__ == '__main__':
    test_data = ['http://127.0.0.1:9090/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg','http://127.0.0.1:9090/A_Comprehensive_Survey_of_Grammar_Error_Correction0001-21.jpg','http://127.0.0.1:9090/2007.158710001-09.jpg','http://127.0.0.1:9090/Relation-Aware_Collaborative_Learning_for_Uni%EF%AC%81ed_Aspect-Based_Sentiment_Analysis0001-02.jpg']
    dopredict(host="127.0.0.1", test_data=test_data)