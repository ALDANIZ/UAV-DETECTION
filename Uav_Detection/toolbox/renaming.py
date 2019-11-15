import cv2
import numpy
import os

liste_1=os.listdir("source")

for i in range(1,5274):   
   m=cv2.imread("source/"+liste_1[i-1])
  
   if(m is None):
      continue
       
   print(str(i+4614)+"."+"Foto"+"----->"+str(m.shape))
      
   cv2.imwrite(str(i+4614)+".jpg",m)




