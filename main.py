#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QFileDialog,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QProgressBar, QPushButton, QVBoxLayout, QWidget)

import imutils


class MyWindow(QWidget):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.makeStatus = 2  # 1表示正在处理视频，2表示不在处理视频
        self.initUI()

    def initUI(self):
        '''初始化UI'''
        # 选择视频文件的布局
        selectFileBox = QHBoxLayout()

        # 提示
        selectFileLabel = QLabel("视频文件: ")

        # 输入框
        self.selectFileLineEdit = QLineEdit()

        # 按钮
        selectFileButton = QPushButton(self)
        selectFileButton.setObjectName("selectFileButton")
        selectFileButton.setText("选择")
        selectFileButton.clicked.connect(self.selectFile)
        # selectFileButton.move(50, 50)

        # 把各个部件添加到布局中
        selectFileBox.addStretch(1)
        selectFileBox.addWidget(selectFileLabel)
        selectFileBox.addWidget(self.selectFileLineEdit)
        selectFileBox.addWidget(selectFileButton)
        selectFileBox.addStretch(1)

        # 选择输出视频路径的布局
        selectDirBox = QHBoxLayout()

        # 提示
        selectDirLabel = QLabel("输出路径: ")

        # 输入框
        self.selectDirLineEdit = QLineEdit()

        # 按钮
        selectDirButton = QPushButton(self)
        selectDirButton.setObjectName("selectDirButton")
        selectDirButton.setText("选取文件夹")
        selectDirButton.clicked.connect(self.selectDir)
        # selectDirButton.move(50, 100)

        # 把各个部件添加到布局中
        selectDirBox.addStretch(1)
        selectDirBox.addWidget(selectDirLabel)
        selectDirBox.addWidget(self.selectDirLineEdit)
        selectDirBox.addWidget(selectDirButton)
        selectDirBox.addStretch(1)

        # 开始按钮布局
        startBox = QHBoxLayout()

        # 按钮
        self.startButton = QPushButton(self)
        self.startButton.setObjectName("startButton")
        self.startButton.setText("开始识别")
        self.startButton.setMinimumWidth(256)
        self.startButton.clicked.connect(self.startRecognition)

        # 把各个部件添加到布局中
        startBox.addStretch(1)
        startBox.addWidget(self.startButton)
        startBox.addStretch(1)

        # 进度条布局
        pbarBox = QHBoxLayout()

        # 进度条
        self.pbar = QProgressBar(self)
        self.pbar.setMinimumWidth(256)
        self.pbar.hide()

        # 把各个部件添加到布局中
        pbarBox.addStretch(1)
        pbarBox.addWidget(self.pbar)
        pbarBox.addStretch(1)

        # 把上面那些布局添加到主布局中
        mainBox = QVBoxLayout()
        mainBox.addStretch(1)
        mainBox.addLayout(selectFileBox)
        mainBox.addLayout(selectDirBox)
        mainBox.addLayout(startBox)
        mainBox.addLayout(pbarBox)
        mainBox.addStretch(1)

        self.setLayout(mainBox)
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('无人机视频流识别')
        # self.setWindowIcon(QIcon('./web.png'))  # 设置窗口的图标
        self.center()

    def center(self):
        '''主窗口居中显示'''
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

    def alert(self, info):
        '''弹出提示'''
        QMessageBox.information(self, " ", info)

    def selectFile(self):
        '''选取单个文件'''
        fileName, filetype = QFileDialog.getOpenFileName(
            self, "选取文件", "", "All Files (*);;")  # 设置文件扩展名过滤,注意用双分号间隔
        self.selectFileLineEdit.setText(fileName)  # 设置输入框的值
        # print(self.selectFileLineEdit.text())  # 打印输入框的值

    def selectDir(self):
        '''选取文件夹'''
        directory = QFileDialog.getExistingDirectory(
            self, "选取文件夹", "D:/pythonCode/")  # 起始路径
        self.selectDirLineEdit.setText(directory)  # 设置输入框的值
        # print(self.selectDirLineEdit.text())  # 打印输入框的值

    def startRecognition(self):
        '''开始处理视频'''

        # 如果是停止识别
        if self.makeStatus == 1:
            # 设置处理状态和按钮文字
            self.makeStatus = 2
            self.startButton.setText("开始识别")
            return

        # 设置处理状态和按钮文字
        self.makeStatus = 1
        self.startButton.setText("停止识别")

        input_video_path = self.selectFileLineEdit.text()  # 输入视频路径
        out_video_path = self.selectDirLineEdit.text()  # 输出路径或者要输出到的文件夹

        # 判断输入视频是否可读
        if not os.access(input_video_path, os.R_OK):
            self.alert("选择的视频文件不可读或不存在!")
            # 设置处理状态和按钮文字
            self.makeStatus = 2
            self.startButton.setText("开始识别")
            return

        # 处理输出路径
        if os.path.isdir(out_video_path):
            # 输出视频路径是文件夹
            dir_name, file_name = os.path.split(
                input_video_path)  # 获取输入视频文件名（带后缀）
            fname, fename = os.path.splitext(file_name)  # 获取输入视频文件名（不带后缀）和后缀名
            file_name = fname + "_detection" + fename  # 新的文件名
            out_video_path = os.path.join(
                out_video_path, file_name)  # 拼接输出视频地址和新的文件名
        elif os.path.isfile(out_video_path):
            # 输出视频路径是文件，判断是否可写
            if not os.access(out_video_path, os.W_OK):
                # 不可写
                self.alert("输出文件不可写!")
                # 设置处理状态和按钮文字
                self.makeStatus = 2
                self.startButton.setText("开始识别")
                return
        else:
            # 不是视频也不是文件，尝试创建
            try:
                file = open(out_video_path, 'w')
                file.close()
            except:
                self.alert("无法创建输出文件!")
                # 设置处理状态和按钮文字
                self.makeStatus = 2
                self.startButton.setText("开始识别")
                return

        # 初始化进度条
        self.step = 1
        self.pbar.setValue(self.step)
        self.pbar.show()

        re = self.makeVideo(input_video_path, out_video_path)
        if re == 1:
            self.alert('处理完成')
        elif re == 2:
            self.alert('已停止')
        else:
            self.alert('处理失败')

        # 设置处理状态和按钮文字
        self.makeStatus = 2
        self.startButton.setText("开始识别")
        # 隐藏进度条
        self.pbar.hide()

    def makeVideo(self, input_video_path, out_video_path):
        '''在原视频找到无人机并生成用红框圈出无人机的视频
        Args:
            input_video_path: 原视频路径
            out_video_path: 输出视频到什么位置
        '''
        # 视频来源
        cap = cv2.VideoCapture(input_video_path)

        # 定义编解码器，创建VideoWriter 对象
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原视频fps
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取原视频尺寸
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 指定输出的视频格式，可以用-1表示选取
        out = cv2.VideoWriter(out_video_path, fourcc, fps, size)
        # 获取视频总帧数
        count_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('\n----------------------\n视频总帧数：    ', count_frame)
        self.pbar.setMaximum(count_frame)  # 设置进度条总长度位视频总帧数
        # 读取正负样本存入list
        pos_dir = './sample/pos/'
        neg_dir = './sample/neg/'
        listPos = []
        listNeg = []
        # 读正样本存入listPos
        for _, _, files in os.walk(pos_dir):
            for f in files:
                pos = cv2.imread(pos_dir + f, 0)
                listPos.append(pos)
        # 读负样本存入listNeg
        for _, _, files in os.walk(neg_dir):
            for f in files:
                neg = cv2.imread(neg_dir + f, 0)
                listNeg.append(neg)


        while cap.isOpened():
            # 如果被停止了（按了停止识别）
            if self.makeStatus == 2:
                return 2

            ok, frame = cap.read()  # 读取一帧数据
            if not ok:
                break

            # 转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # 每隔三帧做一次判断
            if(self.step % 3 == 1):
                firstFrame = gray
            if(self.step % 3 == 2):
                twoFrame = gray
            if(self.step % 3 == 0):
                threeFrame = gray
                # 做帧差
                frameData1 = cv2.absdiff(firstFrame, twoFrame)
                frameData2 = cv2.absdiff(twoFrame, threeFrame)
                # 二进制阈值化
                thresh1 = cv2.threshold(
                    frameData1, 50, 255, cv2.THRESH_BINARY)[1]
                thresh2 = cv2.threshold(
                    frameData2, 50, 255, cv2.THRESH_BINARY)[1]
                # 膨胀
                thresh1 = cv2.dilate(thresh1, None, iterations=4)
                thresh1 = cv2.dilate(thresh2, None, iterations=4)
                result = cv2.bitwise_and(thresh1, thresh2)
                result = cv2.dilate(result, None, iterations=13)
                # 找到图像上的轮廓
                (_, cnts, _) = cv2.findContours(result.copy(),
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 遍历轮廓
                for kk in cnts:
                    # 计算轮廓的边界框，在当前帧中画出该框
                    (x, y, w, h) = cv2.boundingRect(kk)

                    # 对x和y减30，确保范围足够
                    if x > 30:
                        x -= 30
                    else:
                        x = 0
                    if y > 30:
                        y -= 30
                    else:
                        y = 0

                    # 计算xy偏移量
                    if x + w + 60 > size[0]:
                        xw = size[0]
                    else:
                        xw = x + w + 60
                    if y + w + 60 > size[1]:
                        yh = size[1]
                    else:
                        yh = y + w + 60

                    moving = frame[x:xw, y:yh]  # 检测到的移动物体
                    # 如果是空的，就跳过
                    if len(moving) < 1:
                        continue
                    # 判断是否为无人机，如果是，圈出红框
                    if self.IsDrone(moving,listPos,listNeg):
                        cv2.rectangle(frame, (x-30, y-30),
                                            (x + w + 60, y + h + 60), (0, 0, 255), 1)
            cv2.imshow('frame',frame)

            out.write(frame)

            # 更新进度
            self.step += 1
            self.pbar.setValue(self.step)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放摄像头并销毁所有窗口
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return 1

    def IsDrone(self, img,listPos,listNeg):
        ''' 对比传入的图片和样本，返回图片是否为无人机 '''
        # 对比所有正样本取相似度最高值
        posNum = 0
        for f in listPos:
            num = self.match(img, f)
            if posNum < num:
                posNum = num
        # 对比所有负样本取相似度最高值
        negNum = 1
        for f in listNeg:
            num = self.match(img, f)
            if negNum < num:
                negNum = num

        if posNum > negNum:
            return True
        else:
            return False

    def match(self, img1, img2):
        ''' 对比两张图片，返回相似度 '''

        # 使用SIFT检测角点
        sift = cv2.xfeatures2d.SIFT_create()
        # 获取关键点和描述符
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 定义FLANN匹配器
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 使用KNN算法匹配
        matches = flann.knnMatch(des1, des2, k=2)

        # 去除错误匹配
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        return len(good)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
