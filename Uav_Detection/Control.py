import cv2 as cv
import  numpy as np

def Uzaklik_Olcumi(Cozunurluk,Kare_Alani):

    return  Cozunurluk / Kare_Alani
    pass


def Alan_Hesabi(x1,y1,x2,y2):

    return abs((x2-x1) * (y2-y1))
    pass


def Merkezi_Uzunlug_Donme_Acisi(MSatir,MSutun,x1,y1,x2,y2):

        global yatay_aci
        global dikey_aci

        Hedef_Merkez_Nokta_X = ((x2 - x1) // 2) + x1
        Hedef_Merkez_Nokta_Y = ((y2 - y1) // 2) + y1

        dikey_uzunlug = abs(MSatir // 2 - Hedef_Merkez_Nokta_Y)
        yatay_uzunlug = abs(MSutun // 2 - Hedef_Merkez_Nokta_X)


        if dikey_uzunlug == 0 and yatay_uzunlug != 0:
            yatay_aci = np.arctan((dikey_uzunlug / yatay_uzunlug)) * (180 / np.pi)


        elif dikey_uzunlug != 0 and yatay_uzunlug == 0:
            dikey_aci = np.arctan((yatay_uzunlug / dikey_uzunlug)) * (180 / np.pi)

        else:
            yatay_aci = np.arctan((dikey_uzunlug / yatay_uzunlug)) * (180 / np.pi) // 1
            dikey_aci = np.arctan((yatay_uzunlug / dikey_uzunlug)) * (180 / np.pi) // 1

        Merkeze_Uzaklik = np.sqrt(pow(dikey_uzunlug,2) + pow(yatay_uzunlug,2))

        return  yatay_aci,dikey_aci,Merkeze_Uzaklik
        pass





