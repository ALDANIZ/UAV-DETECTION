import numpy as np

x1=[0,0,0,0,0,0,0,0,0,0]
x2=[0,0,0,0,0,0,0,0,0,0]
y1=[0,0,0,0,0,0,0,0,0,0]
y2=[0,0,0,0,0,0,0,0,0,0]

def Birikdir(x_1,y_1,x_2,y_2,frame,limit = 0):

        if frame < limit:
            x1[frame] = x_1
            y1[frame] = y_1
            x2[frame] = x_2
            y2[frame] = y_2
            pass

        else:
            x1[frame % limit] = x_1
            y1[frame % limit] = y_1
            x2[frame % limit] = x_2
            y2[frame % limit] = y_2
            pass

        frame += 1



def Hesapla():

         x1_array = np.asarray(x1)
         y1_array = np.asarray(y1)
         x2_array = np.asarray(x2)
         y2_array = np.asarray(y2)
         x1_ortalama = x1_array.mean()
         y1_ortalama = y1_array.mean()
         x2_ortalama = x2_array.mean()
         y2_ortalama = y2_array.mean()

         return x1_ortalama, y1_ortalama, x2_ortalama, y2_ortalama


def Basari_Yuzdesi(self,Butun_Veri,Istenen_Veri):

        return ( Istenen_Veri / Butun_Veri ) * 100
        pass


