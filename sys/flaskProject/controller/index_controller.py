import functools
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Blueprint, request, redirect, url_for, flash, render_template, session, g, send_file
import sqlite3 as sql

from werkzeug.security import generate_password_hash, check_password_hash

from controller.auth_controller import login_required
from controller.data_pre import data_pree
from controller.a_d import anomaly__detection

index = Blueprint('index', __name__)

filename=[]
anomaly_filename=[]
anomaly_detection_result=[]
columns = ['ip','duration', 'trans_protocol', 'app_protocol', 'send_bytes', 'recv_bytes', 't_same_main_engine',
           't_same_server', 't_same_server_rate', 't_diff_server_rate', 't_diff_main_engine_rate','t_same_main_engine_2',
           't_same_server_2', 't_same_server_rate_2', 't_diff_server_rate_2', 't_diff_main_engine_rate_2',
           'same_main_engine', 'same_server', 'same_server_rate', 'diff_server_rate',
           'same_src_rate', 'diff_src_rate' ,'std','mean','send_rst_flag_rate']
columns2 = ['ip','duration', 'trans_protocol', 'app_protocol', 'send_bytes', 'recv_bytes', 't_same_main_engine',
           't_same_server', 't_same_server_rate', 't_diff_server_rate', 't_diff_main_engine_rate','t_same_main_engine_2',
           't_same_server_2', 't_same_server_rate_2', 't_diff_server_rate_2', 't_diff_main_engine_rate_2',
           'same_main_engine', 'same_server', 'same_server_rate', 'diff_server_rate',
           'same_src_rate', 'diff_src_rate' ,'std','mean','send_rst_flag_rate','label']
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d
@index.route('/manage',methods=('GET','POST'))
@login_required
def manage():
    print("manage")
    con = sql.connect('internet.db')
    con.row_factory = dict_factory
    db = con.cursor()
    users = db.execute(
        'SELECT * FROM user'
    ).fetchall()
    print(users)
    return render_template('index/manage.html',users=users)

@index.route('/modify_user/<user_id>',methods=('GET','POST'))
@login_required
def modify_user(user_id):
    if request.method == 'POST':
        print(user_id)
        username = request.form['name']
        password=request.form['password']
        con = sql.connect('internet.db')
        db = con.cursor()
        db.execute(
            'UPDATE user SET username=?,password=?'
            'where id=?',
            (username, generate_password_hash(password),user_id)
        )
        con.commit()
    return redirect(url_for('index.manage'))

@index.route('/delete_user/<user_id>',methods=('GET','POST'))
@login_required
def delete_user(user_id):
    con = sql.connect('internet.db')
    db = con.cursor()
    db.execute('DELETE FROM user WHERE id=?', (user_id,))
    db.execute('DELETE FROM black WHERE author_id=?', (user_id,))
    db.execute('DELETE FROM white WHERE author_id=?', (user_id,))
    con.commit()
    return redirect(url_for('index.manage'))

@index.route('/white_manage',methods=('GET','POST'))
@login_required
def white_manage():
    print("white_manage")
    con = sql.connect('internet.db')
    con.row_factory = dict_factory
    db = con.cursor()
    whites = db.execute(
        'SELECT * FROM white'
    ).fetchall()
    print(whites)
    return render_template('index/white_manage.html',whites=whites)

@index.route('/modify_manage_white/<ip_id>',methods=('GET','POST'))
@login_required
def modify_manage_white(ip_id):
    if request.method == 'POST':
        print(ip_id)
        ipname = request.form['ipname']
        con = sql.connect('internet.db')
        db = con.cursor()
        db.execute(
            'UPDATE white SET w_text=?'
            'where id=?',
            (ipname, ip_id)
        )
        con.commit()
    return redirect(url_for('index.white_manage'))

@index.route('/delete_manage_white/<ip_id>',methods=('GET','POST'))
@login_required
def delete_manage_white(ip_id):
    con = sql.connect('internet.db')
    db = con.cursor()
    db.execute('DELETE FROM white WHERE id=?', (ip_id,))
    con.commit()
    return redirect(url_for('index.white_manage'))


@index.route('/black_manage',methods=('GET','POST'))
@login_required
def black_manage():
    print("black_manage")
    con = sql.connect('internet.db')
    con.row_factory = dict_factory
    db = con.cursor()
    blacks = db.execute(
        'SELECT * FROM black'
    ).fetchall()
    print(blacks)
    return render_template('index/black_manage.html',blacks=blacks)

@index.route('/modify_manage_black/<ip_id>',methods=('GET','POST'))
@login_required
def modify_manage_black(ip_id):
    if request.method == 'POST':
        print(ip_id)
        ipname = request.form['ipname']
        con = sql.connect('internet.db')
        db = con.cursor()
        db.execute(
            'UPDATE black SET b_text=?'
            'where id=?',
            (ipname, ip_id)
        )
        con.commit()
    return redirect(url_for('index.black_manage'))

@index.route('/delete_manage_black/<ip_id>',methods=('GET','POST'))
@login_required
def delete_manage_black(ip_id):
    con = sql.connect('internet.db')
    db = con.cursor()
    db.execute('DELETE FROM black WHERE id=?', (ip_id,))
    con.commit()
    return redirect(url_for('index.black_manage'))


@index.route('/shouye1',methods=('GET','POST'))
@login_required
def shouye1():
    print("shouye1")
    return render_template('index/shouye1.html')


@index.route('/shouye',methods=('GET','POST'))
@login_required
def shouye():
    print("a")
    return render_template('index/shouye.html')

@index.route('/white',methods=('GET','POST'))
@login_required
def white():
    user_id=session.get('user_id')
    con = sql.connect('internet.db')
    con.row_factory = dict_factory
    db = con.cursor()
    white_datas = db.execute(
        'SELECT * FROM white WHERE author_id = ?', (user_id,)
    ).fetchall()
    print("c")
    print(white_datas)
    return render_template('index/white.html', white_datas=white_datas)


@index.route('/black',methods=('GET','POST'))
@login_required
def black():
    uesr_id=session.get('user_id')
    con = sql.connect('internet.db')
    con.row_factory = dict_factory
    db = con.cursor()
    black_datas = db.execute(
        'SELECT * FROM black WHERE author_id = ?', (uesr_id,)
    ).fetchall()
    print("c")
    print(black_datas)
    return render_template('index/black.html',black_datas=black_datas)

@index.route('/error',methods=('GET','POST'))
@login_required
def error():
    print("d")
    return render_template('index/error.html')

@index.route('/pic',methods=('GET','POST'))
@login_required
def pic():
    print("e")
    return render_template('index/pic.html')

@index.route('/upload',methods=('GET','POST'))
@login_required
def upload():
    if request.method == 'POST':  # 如果请求类型为POST，说明是文件上传请求
        f = request.files.get('file')  # 获取文件对象
        f.save(os.path.join('static', f.filename))  # 保存文件
        filename.append(f.filename)
    return render_template('index/shouye.html')

@index.route('/transform',methods=('GET','POST'))
@login_required
def transform():
    if request.method == 'POST':
        file=filename[-1]
        result=data_pree(file)
        print(result)
        result_data=pd.DataFrame(result,columns=columns)
        name='_result'
        result_data.to_csv("static/%s%s.csv"%(file,name), sep=',', index=False)
        fff='%s%s.csv'%(file,name)
    return render_template('index/shouye.html',data=fff)

@index.route('/download/<data>',methods=('GET','POST'))
@login_required
def download(data):
    return send_file('static/%s'%(data), as_attachment=True)

@index.route('/download_file1',methods=('GET','POST'))
@login_required
def download_file1():
    return send_file('static/or_data.csv', as_attachment=True)

@index.route('/download_file2',methods=('GET','POST'))
@login_required
def download_file2():
    return send_file('static/or_data.csv_result.csv', as_attachment=True)

@index.route('/modify/<ip_id>' ,methods=('GET','POST'))
@login_required
def modify(ip_id):
    if request.method=='POST':
        text=request.form['ipname']
        con=sql.connect('internet.db')
        db=con.cursor()
        db.execute(
            'UPDATE black SET b_text=?'
            'where id=?',
            (text,ip_id)
        )
        con.commit()
    return redirect(url_for('index.black'))

@index.route('/delete/<ip_id>',methods=('GET','POST'))
@login_required
def delete(ip_id):
    con=sql.connect('internet.db')
    db=con.cursor()
    db.execute('DELETE FROM black WHERE id=?',(ip_id,))
    con.commit()
    return redirect(url_for('index.black'))

@index.route('/add',methods=('GET','POST'))
@login_required
def add():
    if request.method =='POST':
        uesr_id = session.get('user_id')
        text=request.form['ipname']
        con=sql.connect('internet.db')
        db=con.cursor()
        db.execute(
            "INSERT INTO black(author_id,b_text) VALUES (?,?)",
            (uesr_id,text),
        )
        con.commit()
    return redirect(url_for('index.black'))

@index.route('/modify_white/<ip_id>' ,methods=('GET','POST'))
@login_required
def modify_white(ip_id):
    if request.method=='POST':
        text=request.form['ipname']
        con=sql.connect('internet.db')
        db=con.cursor()
        db.execute(
            'UPDATE white SET w_text=?'
            'where id=?',
            (text,ip_id)
        )
        con.commit()
    return redirect(url_for('index.white'))

@index.route('/delete_white/<ip_id>',methods=('GET','POST'))
@login_required
def delete_white(ip_id):
    con=sql.connect('internet.db')
    db=con.cursor()
    db.execute('DELETE FROM white WHERE id=?',(ip_id,))
    con.commit()
    return redirect(url_for('index.white'))

@index.route('/add_white',methods=('GET','POST'))
@login_required
def add_white():
    if request.method =='POST':
        uesr_id = session.get('user_id')
        text=request.form['ipname']
        con=sql.connect('internet.db')
        db=con.cursor()
        db.execute(
            "INSERT INTO white(author_id,w_text) VALUES (?,?)",
            (uesr_id,text),
        )
        con.commit()
    return redirect(url_for('index.white'))

@index.route('/anomaly_detection',methods=('GET','POST'))
@login_required
def anomaly_detection():
    if request.method == 'POST':  # 如果请求类型为POST，说明是文件上传请求
        f = request.files.get('file')  # 获取文件对象
        f.save(os.path.join('static', f.filename))  # 保存文件
        anomaly_filename.append(f.filename)
    return render_template('index/error.html')

@index.route('/error_ad',methods=('GET','POST'))
@login_required
def error_ad():
    if request.method == 'POST':
        file=anomaly_filename[-1]
        uesr_id = session.get('user_id')
        con = sql.connect('internet.db')
        con.row_factory = dict_factory
        db = con.cursor()
        blacks = db.execute(
            'SELECT b_text FROM black WHERE author_id = ?', (uesr_id,)
        ).fetchall()
        whites=db.execute(
            'SELECT w_text FROM white WHERE author_id = ?', (uesr_id,)
        )
        black_ip=[]
        white_ip=[]
        for black in blacks:
            black_ip.append(black['b_text'])
        for white in whites:
            white_ip.append(white['w_text'])
        result=anomaly__detection(file,black_ip,white_ip)
        print(result)
        result_data=pd.DataFrame(result,columns=columns2)
        name='_result'
        result_data.to_csv("static/%s%s.csv"%(file,name), sep=',', index=False)
        fff='%s%s.csv'%(file,name)
        anomaly_detection_result.append(fff)
    return render_template('index/error.html',data=fff)

@index.route('/download_ad/<data>',methods=('GET','POST'))
@login_required
def download_ad(data):
    return send_file('static/%s'%(data), as_attachment=True)


@index.route('/draw_app',methods=('GET','POST'))
@login_required
def draw_app():
    if request.method == 'POST':
        file=filename[-1]
        df = pd.read_csv("static/%s" % (file))
        data_analysis = []
        app_protocol_list = ['HTTP', 'SSL', 'HTTPS', 'DNS', 'SAMBA', 'MYSQL', 'OTHER']
        app_protocol = df['app_protocol'].tolist()
        HTTP = app_protocol.count('HTTP')
        data_analysis.append(HTTP)
        SSL = app_protocol.count('SSL')
        data_analysis.append(SSL)
        HTTPS = app_protocol.count('HTTPS')
        data_analysis.append(HTTPS)
        DNS = app_protocol.count('DNS')
        data_analysis.append(DNS)
        SAMBA = app_protocol.count('SAMBA')
        data_analysis.append(SAMBA)
        MYSQL = app_protocol.count('MYSQL')
        data_analysis.append(MYSQL)
        OTHER = app_protocol.count('OTHER')
        data_analysis.append(OTHER)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2 - 0.16, 1.01 * height, '%s' % int(height), size=10,
                         family="Times new roman")

        x_data = app_protocol_list
        y_data = data_analysis

        #plt.rcParams["font.sans-serif"] = ["SimHei"]
        #plt.rcParams["axes.unicode_minus"] = False

        # 画图，plt.bar()可以画柱状图
        cm = plt.bar(x_data, y_data)
        autolabel(cm)
        # for i in range(len(x_data)):
        # plt.bar(x_data[i], y_data[i])

        # 设置图片名称
        plt.title("应用层协议统计")
        # 设置x轴标签名
        plt.xlabel("应用层协议")
        # 设置y轴标签名
        plt.ylabel("数量")
        # 显示
        name = 'app_protocol.png'
        plt.savefig("static/%s" % (name))
        plt.show()
    return render_template('index/pic.html',data=name)

@index.route('/draw_ad',methods=('GET','POST'))
@login_required
def draw_ad():
    if request.method == 'POST':
        file=anomaly_detection_result[-1]
        df = pd.read_csv("static/%s" % (file))
        ad = []
        anomaly_detection_list = ['正常流量', '异常流量']
        anomaly_detection = df['label'].tolist()
        normal = anomaly_detection.count(0)
        ad.append(normal)
        anomaly = anomaly_detection.count(1)
        ad.append(anomaly)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2 - 0.06, 1.01 * height, '%s' % int(height), size=10,
                         family="Times new roman")

        x_data = anomaly_detection_list
        y_data = ad

        #plt.rcParams["font.sans-serif"] = ["SimHei"]
        #plt.rcParams["axes.unicode_minus"] = False

        # 画图，plt.bar()可以画柱状图
        cm = plt.bar(x_data, y_data,width=0.2)
        autolabel(cm)
        # for i in range(len(x_data)):
        # plt.bar(x_data[i], y_data[i])

        # 设置图片名称
        plt.title("异常检测情况统计")
        # 设置x轴标签名
        plt.xlabel("网络流量类型")
        # 设置y轴标签名
        plt.ylabel("数量")
        # 显示
        name = 'ad.png'
        plt.savefig("static/%s" % (name))
        plt.show()
    return render_template('index/pic.html',data=name)

@index.route('/draw_bw',methods=('GET','POST'))
@login_required
def draw_bw():
    if request.method == 'POST':
        uesr_id = session.get('user_id')
        con = sql.connect('internet.db')
        con.row_factory = dict_factory
        db = con.cursor()
        blacks = db.execute(
            'SELECT b_text FROM black WHERE author_id = ?', (uesr_id,)
        ).fetchall()
        whites = db.execute(
            'SELECT w_text FROM white WHERE author_id = ?', (uesr_id,)
        )
        black_ip = []
        white_ip = []
        for black in blacks:
            black_ip.append(black['b_text'])
        for white in whites:
            white_ip.append(white['w_text'])
        file = anomaly_detection_result[-1]
        df = pd.read_csv("static/%s" % (file))
        #plt.rcParams["font.sans-serif"] = ["SimHei"]
        #plt.rcParams["axes.unicode_minus"] = False
        black_white = df['ip'].tolist()
        black_num = []
        white_num = []
        for i in range(len(black_ip)):
            a = black_white.count(black_ip[i])
            black_num.append(a)

        for k in range(len(white_ip)):
            a = black_white.count(white_ip[k])
            white_num.append(a)

        #plt.style.use('fivethirtyeight')
        plt.pie(black_num, labels=black_ip,autopct='%1.1f%%')
        plt.title('黑名单各IP占比')
        name = 'black.png'
        plt.savefig("static/%s" % (name))
        plt.show()
        plt.pie(white_num, labels=white_ip,autopct='%1.1f%%')
        plt.title('白名单各IP占比')
        name1 = 'white.png'
        plt.savefig("static/%s" % (name1))
        plt.show()
    return render_template('index/pic.html',data=name,data1=name1)