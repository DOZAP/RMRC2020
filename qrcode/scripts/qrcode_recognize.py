#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rospy
import cv2
from pyzbar.pyzbar import decode
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError


#------------------------------------------------------------
#
#Main Class
#
#------------------------------------------------------------
class QRcodeRecognize:
    def __init__(self):
        self.image_pub = rospy.Publisher("qrcode",Image,queue_size=5)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image_raw",Image,self.callback)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def callback(self,data):
        """uvc_camからpublishされたものをOpenCVで扱える形式にconvert"""
        try:
            cv_cam = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)
        
        """ここから画像に対する処理"""
        cv_result = cv_cam.copy()
	decoded_data = decode(cv_cam)
        if decoded_data:
            for qrcode in decoded_data:
                x,y,w,h =qrcode.rect
                cv2.rectangle(cv_result,(x,y),(x+w,y+h),(0,0,255),2)
                qrcodeData = qrcode.data.decode('utf-8')
                cv_result = cv2.putText(cv_result,qrcodeData,(x,y-10),self.font,.5,(0,0,255),2,cv2.LINE_AA) 
        """ここまで画像に対する処理"""

        """OpenCVの形式の画像をROSで扱える形式にconvert"""
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_result,"bgr8"))
        except CvBridgeError,e:
            print(e)


#------------------------------------------------------------
#
#Main Loop
#
#------------------------------------------------------------
def main(args):
    ic = QRcodeRecognize()
    rospy.init_node("qrcode_node",anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destoryAllWindows()


if __name__ == '__main__':
    main(sys.argv)
