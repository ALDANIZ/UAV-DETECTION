import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import math
import matplotlib.pyplot as plt
from Control import*
from Statistics import*
from Kalman import Tahmin

sys.path.append("..")

PATH_TO_CKPT = 'models/Uav/frozen_inference_graph_SSD.pb'
PATH_TO_VIDEO = 'test/test.mp4'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

frame_count = 0
xMax,yMax  = 1000,600
alanMax = xMax * yMax
xOrta,yOrta = xMax // 2,yMax // 2
mavi_x1,mavi_y1 = xMax // 4,yMax // 10
mavi_x2 ,mavi_y2 = xMax - xMax // 4,yMax - yMax // 10
PredictedCoords_1 = np.zeros((2, 1), np.float32)
PredictedCoords_2 = np.zeros((2, 1), np.float32)
xmin ,xmax ,ymin ,ymax,toplam,basari= 0,0,0,0,0,0

video = cv2.VideoCapture(PATH_TO_VIDEO)

while(video.isOpened()):

    ret, frame = video.read()
    h, w, c = frame.shape
    frame_expanded = np.expand_dims(frame, axis=0)
    frame = cv2.resize(frame, (xMax, yMax))

    cv2.rectangle(frame, (mavi_x1, mavi_y1), (mavi_x2, mavi_y2), (255, 0, 0), 2)
    cv2.line(frame, (0, yOrta), (xMax, yOrta), (0, 0, 255), 1)
    cv2.line(frame, (xOrta, 0), (xOrta, yMax), (0, 0, 255), 1)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes,
                                              num_detections],feed_dict={image_tensor: frame_expanded})

    if scores[0, 0] >= 0.3 and classes[0,0] == 1:

        ymin = int((boxes[0][0][0] * yMax))
        xmin = int((boxes[0][0][1] * xMax))
        ymax = int((boxes[0][0][2] * yMax))
        xmax = int((boxes[0][0][3] * xMax))

        merkez_x = ((xmax - xmin) // 2) + xmin
        merkez_y = ((ymax - ymin) // 2) + ymin

        Birikdir(xmin, ymin, xmax, ymax, frame_count, limit=10)

        frame_count += 1
        predictedCoords_1 = Tahmin(xmin, ymin)
        predictedCoords_2 = Tahmin(xmax, ymax)

        kalman_merkez_x = abs(((predictedCoords_2[0] - predictedCoords_1[0])) // 2) + predictedCoords_1[0]
        kalman_merkez_y = abs(((predictedCoords_2[1] - predictedCoords_1[1])) // 2) + predictedCoords_1[1]


        basari += 1

    else:
        x_1_Ortalama, y_1_Ortalama, x_2_Ortalama, y_2_Ortalama = Hesapla()
        predictedCoords_1 = Tahmin(x_1_Ortalama, y_1_Ortalama)
        predictedCoords_2 = Tahmin(x_2_Ortalama, y_2_Ortalama)

    uzaklik_katsayisi = int(Uzaklik_Olcumi(Cozunurluk=xMax * yMax,Kare_Alani=Alan_Hesabi(xmin, ymin, xmax, ymax)))

    kalman_merkez_x = abs(((predictedCoords_2[0] - predictedCoords_1[0])) // 2) + predictedCoords_1[0]
    kalman_merkez_y = abs(((predictedCoords_2[1] - predictedCoords_1[1])) // 2) + predictedCoords_1[1]

    cv2.rectangle(frame, (predictedCoords_1[0], predictedCoords_1[1]), (predictedCoords_2[0], predictedCoords_2[1]),(0, 0, 255), 2)
    cv2.line(frame, (xOrta, yOrta), (kalman_merkez_x, kalman_merkez_y), (255, 0, 0), 2)
    cv2.circle(frame, (int(predictedCoords_1[0]), int(predictedCoords_1[1])), 5, (0, 0, 0), -1)
    cv2.circle(frame, (int(predictedCoords_2[0]), int(predictedCoords_2[1])), 5, (0, 0, 0), -1)

    yatay_Aci, dikey_Aci, Merkez_Uzunlug = Merkezi_Uzunlug_Donme_Acisi(yMax, xMax, xmin, ymin, xmax, ymax)

    cv2.putText(frame, "Uzaklik Katsayi:" + str(uzaklik_katsayisi), (xMax - 250, yMax - 230),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_AA)


    if kalman_merkez_x >= xOrta and kalman_merkez_y <= yOrta:
        cv2.putText(frame, "Bolge : 1", (xMax - 250, yMax - 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1,
                    cv2.LINE_4)
        cv2.putText(frame, "Donus yukari : " + str(yatay_Aci), (xMax - 250, yMax - 170), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 255, 255), 1, cv2.LINE_4)
        cv2.putText(frame, "Donus saga:" + str(dikey_Aci), (xMax - 250, yMax - 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (0, 255, 255), 1, cv2.LINE_4)


    elif kalman_merkez_x <= xOrta and kalman_merkez_y <= yOrta:
        cv2.putText(frame, "Bolge 2", (xMax - 250, yMax - 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1,
                    cv2.LINE_4)
        cv2.putText(frame, "Donus yukari:" + str(yatay_Aci), (xMax - 250, yMax - 170), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 255, 255), 1, cv2.LINE_4)
        cv2.putText(frame, "Donus sola:" + str(dikey_Aci), (xMax - 250, yMax - 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (0, 255, 255), 1, cv2.LINE_4)


    elif kalman_merkez_x < xOrta and kalman_merkez_y >= yOrta:
        cv2.putText(frame, "Bolge 3", (xMax - 250, yMax - 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1,
                    cv2.LINE_4)
        cv2.putText(frame, "Donus asagi:" + str(yatay_Aci), (xMax - 250, yMax - 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (0, 255, 255), 1, cv2.LINE_4)
        cv2.putText(frame, "Donus sola:" + str(dikey_Aci), (xMax - 250, yMax - 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (0, 255, 255), 1, cv2.LINE_4)


    elif kalman_merkez_x > xOrta and kalman_merkez_y >= yOrta:
        cv2.putText(frame, "Bolge 4", (xMax - 250, yMax - 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1,
                    cv2.LINE_4)
        cv2.putText(frame, "Donus asagi:" + str(yatay_Aci), (xMax - 250, yMax - 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (0, 255, 255), 1, cv2.LINE_4)
        cv2.putText(frame, "Donus saga :" + str(dikey_Aci), (xMax - 250, yMax - 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (0, 255, 255), 1, cv2.LINE_4)

    cv2.putText(frame, "Sol Konum:" + str(int(predictedCoords_2[0])) + "," + str(int(predictedCoords_2[1])),
                (xMax - 250, yMax - 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_4)
    cv2.putText(frame, "Sag Konum:" + str(int(predictedCoords_1[0])) + "," + str(int(predictedCoords_1[1])),
                (xMax - 250, yMax - 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_4)

    cv2.imshow('Takip sistemi', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
