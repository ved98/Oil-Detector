
# coding: utf-8

# In[ ]:


from keras.layers import Convolution2D, MaxPooling2D , BatchNormalization, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense , Input
from keras.models import Model, Sequential
from keras import applications, losses
from numpy import *
import cv2
import numpy as np
import scipy.misc
import os
from keras.optimizers import Adadelta, RMSprop, adam, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical


# In[ ]:


neck = applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


classify = Flatten()(neck.output)
classify = Dense(256, activation='relu')(classify)
classify = Dropout(0.5)(classify)
classify = Dense(2, activation='sigmoid')(classify)


# In[ ]:


Detector=Model(neck.input, classify)


# In[ ]:


Detector.summary()


# In[ ]:


Detector.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


train_datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2, rescale=1./255,
        horizontal_flip=True,
        fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale = 1./255)

# In[ ]:


pimgs=os.listdir('data2/positive')
nimgs=os.listdir('data2/negative')


# In[ ]:


in_mat=[]
out_mat=[]


# In[ ]:


for i in range(len(pimgs)):
    img = load_img('data2/positive/'+pimgs[i])
    img = img_to_array(img)
    in_mat.append(img)
    out_mat.append(1)
print(str(len(pimgs))+" positives")
for i in range(len(nimgs)):
    img = load_img('data2/negative/'+nimgs[i])
    img = img_to_array(img)
    in_mat.append(img)
    out_mat.append(0)
print(str(len(nimgs))+" negatives")
print(str(len(pimgs)+len(nimgs))+" images are loaded")


# In[ ]:


data,Label = shuffle(in_mat,out_mat, random_state=2)
X_train,X_test,Y_train,Y_test=train_test_split(data, Label, test_size=0.10, random_state=2)
del data[:]
del Label[:]
del in_mat[:]
del out_mat[:]
X_train=array(X_train)
Y_train=array(Y_train)
X_test=array(X_test)
Y_test=array(Y_test)
Y_train= to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)
print("traindata = "+str(X_train.shape[0]))
print("testdata = "+str(X_test.shape[0]))


# In[ ]:
train_gen = train_datagen.flow(X_train, Y_train, batch_size=4)
test_gen= test_datagen.flow(X_test , Y_test, batch_size=4)
#datagen.fit(X_train)
print("Training - Part 1 started!!")
Detector.fit_generator(train_gen, steps_per_epoch=3000, validation_data=test_gen, validation_steps=X_test.shape[0]/4, epochs=50, verbose=1)
# for epoch in range(1):
#     X_train,Y_train=shuffle(X_train,Y_train)
#     print ("Epoch is: %d\n" % epoch)
#     print ("Number of batches: %d\n" % int(num_batches))
#     for batch in range(int(num_batches)):
#         batch_train_X=X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
#         batch_train_Y=Y_train[batch*batch_size:min((batch+1)*batch_size,len(Y_train))]
#         loss=Detector.train_on_batch(batch_train_X,batch_train_Y)
# #         l=Detector.predict(batch_train_X,verbose=0)
# #         loss=binary_crossentropy(batch_train_Y,l)
# #         print(loss)
#         print ('epoch_num: %d batch_num: %d train loss: %f class-%f\n' % (epoch+1,batch+1,loss[0]*100,loss[1]))
Detector.save_weights("new_weights.h5")
print("PART1 - training completed for the classifier")


# In[ ]:


scores = Detector.evaluate(X_test, Y_test, verbose=1)
print ("Test Error: %.2f%%" % (100-scores[1]*100))
print ("Test Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


# FDetector=Model(neck.input,classify)


# # In[ ]:


# i=1
# for layers in neck.layers:
#     layers.trainable=False
#     if i == 20:
#         break
#     i+=1


# # In[ ]:


# FDetector.summary()


# # In[ ]:


# FDetector.load_weights('part1.h5')


# # In[ ]:


# FDetector.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# # In[ ]:


# print("Training - Part 2 started")
# batch_size=1
# FDetector.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=1)
# # num_batches=math.ceil(len(X_train)*1.0/batch_size)
# # for epoch in range(1):
# #     X_train,Y_train=shuffle(X_train,Y_train)
# #     print ("Epoch is: %d\n" % epoch)
# #     print ("Number of batches: %d\n" % int(num_batches))
# #     for batch in range(int(num_batches-40)):
# #         batch_train_X=X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
# #         batch_train_Y=Y_train[batch*batch_size:min((batch+1)*batch_size,len(Y_train))]
# #         loss=FDetector.train_on_batch(batch_train_X,batch_train_Y)
# # #         l=Detector.predict(batch_train_X,verbose=0)
# # #         loss=binary_crossentropy(batch_train_Y,l)
# # #         print(loss)
# #         print ('epoch_num: %d batch_num: %d train loss: %f class-%f\n' % (epoch+1,batch+1,loss[0]*100,loss[1]))
# FDetector.save_weights("detector.h5")
# print("PART2 - training completed for the network")


# # In[ ]:


# scores = FDetector.evaluate(X_test, Y_test, verbose=1)
# print ("Test Error: %.2f%%" % (100-scores[1]*100))
# print ("Test Accuracy: %.2f%%" % (scores[1]*100))

