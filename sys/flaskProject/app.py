from flask import Flask, render_template, redirect, url_for,jsonify,request
import sqlite3
from flask_bootstrap import Bootstrap
from flask_dropzone import Dropzone
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import heapq
import random
from sklearn.preprocessing import MinMaxScaler
#from gevent import pywsgi

from controller.auth_controller import auth
from controller.index_controller import index


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32,output_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        decoder = self.sigmoid(out)
        return decoder

app = Flask(__name__)
bootstrap = Bootstrap(app)
dropzone=Dropzone(app)
app.secret_key='jkjiji'
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = '.csv'
app.config['DROPZONE_MAX_FILE_SIZE']=10
app.config['DROPZONE_DEFAULT_MESSAGE']='拖拽或点击上传csv文件'
app.register_blueprint(auth, url_prefix='/anomaly-detection')
app.register_blueprint(index,url_prefix='/anomaly-detection/index')
#app.config['DROPZONE_REDIRECT_VIEW']='index.shouye'


if __name__ == '__main__':
    #server=pywsgi.WSGIServer(('0.0.0.0',5000),app)
    #server.serve_forever()
    app.run(host='0.0.0.0',port=5003)
