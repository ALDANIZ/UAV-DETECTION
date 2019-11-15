import cv2
import os
import numpy

liste=os.listdir("source")

for i in range(0,len(liste)):

   src = cv2.imread("source/"+liste[i])
   #scale_percent = 50
   #width = int(src.shape[1] * scale_percent / 70)
   #height = int(src.shape[0] * scale_percent / 70)
   #dsize = (width, height)

   output = cv2.resize(src,(300,300))

   cv2.imwrite("300_X_300/" + liste[i],output)

   cv2.waitKey(0)

   print("foto:"+str(6000+i)+"------>"+str(output.shape)) 

