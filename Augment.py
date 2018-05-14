
# coding: utf-8

# In[38]:


#import the libraries
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
import numpy as np


# In[39]:


#function to resize the image for the CNN
def centering_image(img):
    size = [256,256]
    
    img_size = img.shape[:2]
    
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized


# In[40]:


#datagenerator
datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# In[41]:


pimgs=os.listdir('data/positive')
nimgs=os.listdir('data/negative')


# In[ ]:


print(str(len(pimgs))+" postive images are found ")
print(str(len(nimgs))+" negative images are found ")


# In[42]:


for i in range(len(pimgs)):
    img = load_img('data/positive/'+pimgs[i])
    x = img_to_array(img)
    if(x.shape[0] > x.shape[1]):
        tile_size = (int(x.shape[1]*256/x.shape[0]),256)
    else:
        tile_size = (256, int(x.shape[0]*256/x.shape[1]))
    x = centering_image(cv2.resize(x, dsize=tile_size))
    x = x[16:240, 16:240]
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
    j = 0
    for b in datagen.flow(x, batch_size=1,save_to_dir='data2/positive', save_prefix=pimgs[i], save_format='jpeg'):
        j += 1
        if j > 10:
            break # otherwise the generator would loop indefinitely
print("Positive images augmented and saved")


# In[43]:


for i in range(len(nimgs)):
    img = load_img('data/negative/'+nimgs[i])
    x = img_to_array(img)
    if(x.shape[0] > x.shape[1]):
        tile_size = (int(x.shape[1]*256/x.shape[0]),256)
    else:
        tile_size = (256, int(x.shape[0]*256/x.shape[1]))
    x = centering_image(cv2.resize(x, dsize=tile_size))
    x = x[16:240, 16:240]
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
    j = 0
    for b in datagen.flow(x, batch_size=1,save_to_dir='data2/negative', save_prefix=nimgs[i], save_format='jpeg'):
        j += 1
        if j > 20:
            break # otherwise the generator would loop indefinitely
print("Negative images augmented and saved")

