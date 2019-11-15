from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

datagen=ImageDataGenerator(

    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.1,
    horizontal_flip=0.1,
    brightness_range=(0.8,1.3),
    shear_range=0.2,
    channel_shift_range=0.2)

numaralandirma=-1

for i in range(1,156):

    numaralandirma+=1

    path ="source/"+str(i+6000)+".jpg"

    img = image.load_img(path);

    img = image.img_to_array(img)

    input_batch = img.reshape((1, *img.shape))

    j = 1;

    print("  Foto" + " "+str(i+6000)+" "+ "islenir")

    for batch in datagen.flow(input_batch,batch_size=1):
      
      a=image.img_to_array(batch[0])/255.0

      print("            islenme sayisi:"+str(j)+"-------->"+str(a.shape))

      numaralandirma+=1

      plt.imsave(str(numaralandirma+6000)+".jpg",a)

      j+=1

      if j==9:

          break
      

