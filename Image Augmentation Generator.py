import tensorflow as tf
import random 
from PIL import Image


def imageGenerator(k=1000,store_loc="output_generated_set",og_loc='python_file_set'):
    rotationGenerator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40)
    flipGenerator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
    channelGenerator = tf.keras.preprocessing.image.ImageDataGenerator(channel_shift_range=100)
    zoomGenerator = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.5,2])
    shearGenerator = tf.keras.preprocessing.image.ImageDataGenerator(shear_range=40)
    brightnessGenerator = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=(.15,2.5))
    dimGenerator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=[-100,-50,0,50,100],height_shift_range=[-50,0,50])
    l=[rotationGenerator,flipGenerator,channelGenerator,zoomGenerator,shearGenerator,brightnessGenerator,dimGenerator]
    for i in range(1,k+1):
        t=random.choice(l)
        x, y = next(t.flow_from_directory(og_loc, batch_size=1))
        iName=store_loc+"\\generatedPic"+str(i)+".jpg"
        img=Image.fromarray(x[0].astype('uint8'))
        img.save(iName,'JPEG')
        
#imageGenerator()        
        
        