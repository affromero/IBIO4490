#coding: utf-8
import sys
import os
sys.path.append('/home/afromero/caffe-master/python')
os.environ['CUDA_VISIBLE_DEVICES']='0'
import caffe
caffe.set_mode_gpu()

prototxt = 'val.prototxt'
net = caffe.Net(prototxt, 1)
