#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError


#------------------------------------------------------------
#
#Constant Definition
#
#------------------------------------------------------------
TEMPLATE_IMAGES_PATH = "~/catkin_ws/src/RMRC2020/hazmat/hazmat_label2020/"

MIN_MATCH_COUNT = 80

MIN_AREA = 32000

MATCHING_RATIO = 0.6


#------------------------------------------------------------
#
#Import Template Images
#
#------------------------------------------------------------
hazmat_list = []
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-1.3-explosive.png"),"1.3 Explosives"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-1.4-explosive.png"),"1.4 Explosives"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-1.5-blasting-agent.png"),"1.5 Blasting Agent"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-2-flammable-gas.png"),"Flammable Gas"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-2-poison-gas.png"),"Poison Gas"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-3-flammable-liquid.png"),"Flammable Liquid"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-4-dangerous-when-wet.png"),"Dangerous When Wet"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-4-spontaneously-combustible.png"),"Spontaneously Combustible"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-5.1-oxidizer.png"),"Oxidizer"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-6-infectious-substance.png"),"Infectious Substance"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-6-poison-inhalation-hazard.png"),"Inhalation Hazard"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-6-poison.png"),"Poison"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-7-radioactive-ii.png"),"Radioactive ii"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-7-radioactive-iii.png"),"Radioactive iii"))
hazmat_list.append((cv2.imread(TEMPLATE_IMAGES_PATH+"label-8-corrosive.png"),"Corrosive"))


#------------------------------------------------------------
#
#Preparation
#
#------------------------------------------------------------
sift = cv2.xfeatures2d.SIFT_create()

kp1 , des1 = sift.detectAndCompute(hazmat_list[0][0] ,None)
kp2 , des2 = sift.detectAndCompute(hazmat_list[1][0] ,None)
kp3 , des3 = sift.detectAndCompute(hazmat_list[2][0] ,None)
kp4 , des4 = sift.detectAndCompute(hazmat_list[3][0] ,None)
kp5 , des5 = sift.detectAndCompute(hazmat_list[4][0] ,None)
kp6 , des6 = sift.detectAndCompute(hazmat_list[5][0] ,None)
kp7 , des7 = sift.detectAndCompute(hazmat_list[6][0] ,None)
kp8 , des8 = sift.detectAndCompute(hazmat_list[7][0] ,None)
kp9 , des9 = sift.detectAndCompute(hazmat_list[8][0] ,None)
kp10,des10 = sift.detectAndCompute(hazmat_list[9][0] ,None)
kp11,des11 = sift.detectAndCompute(hazmat_list[10][0],None)
kp12,des12 = sift.detectAndCompute(hazmat_list[11][0],None)
kp13,des13 = sift.detectAndCompute(hazmat_list[12][0],None)
kp14,des14 = sift.detectAndCompute(hazmat_list[13][0],None)
kp15,des15 = sift.detectAndCompute(hazmat_list[14][0],None)

kp_des_list = []
kp_des_list.append((kp1 ,des1))
kp_des_list.append((kp2 ,des2))
kp_des_list.append((kp3 ,des3))
kp_des_list.append((kp4 ,des4))
kp_des_list.append((kp5 ,des5))
kp_des_list.append((kp6 ,des6))
kp_des_list.append((kp7 ,des7))
kp_des_list.append((kp8 ,des8))
kp_des_list.append((kp9 ,des9))
kp_des_list.append((kp10,des10))
kp_des_list.append((kp11,des11))
kp_des_list.append((kp12,des12))
kp_des_list.append((kp13,des13))
kp_des_list.append((kp14,des14))
kp_des_list.append((kp15,des15))


#------------------------------------------------------------
#
#Main Class
#
#------------------------------------------------------------
class DetectAndRecognize:
    def __init__(self):
        self.image_pub = rospy.Publisher("hazmat",Image,queue_size=5)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image_raw",Image,self.callback)

    def callback(self,data):
        """uvc_camからpublishされたものをOpenCVで扱える形式にconvert"""
        try:
            cv_cam = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)
        
        """ここから画像に対する処理"""
        cv_result = cv_cam.copy()
        kp_cam,des_cam = sift.detectAndCompute(cv_cam,None)
        for i,(kp_temp,des_temp) in enumerate(kp_des_list):
            """特徴量のマッチング"""
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_temp,des_cam,k=2)
            good = []
            for m,n in matches:
                if m.distance < MATCHING_RATIO * n.distance:
                    good.append(m)
            """平面の推定"""
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp_cam[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                h,w = 1188,1188   #h,w = temp_img.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                #print(cv2.contourArea(np.int32(dst))) 面積デバッグ用
                """面積の大きさ制限の処理"""
                if cv2.contourArea(np.int32(dst)) < MIN_AREA:
                    continue
                """矩形と文字を描画"""
                cv_result = cv2.polylines(cv_result,[np.int32(dst)],True,(0,255,0),8,cv2.LINE_AA)
                cv_result = cv2.putText(cv_result,hazmat_list[i][1],tuple(dst[1,0]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),thickness=3)
            else:
                matchesMask = None

            """キーポイントを先で結ぶ処理(普段はコメントアウト)"""
            #draw_params = dict(matchColor = (0,255,0), #matching line color (green)
            #                   singlePointColor = None,
            #                   matchesMask = matchesMask, #only inliers
            #                   flags = 2)
            #cv_result = cv2.drawMatches(hazmat_list[i][0],kp_temp,cv_cam,kp_cam,good,None,**draw_params)
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
    ic = DetectAndRecognize()
    rospy.init_node("hazmat_node",anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
        cv2.destoryAllWindows()


if __name__ == '__main__':
    main(sys.argv)

