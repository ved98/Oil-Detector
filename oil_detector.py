
# coding: utf-8

# In[ ]:


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


# In[ ]:


neck = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


classify = Flatten()(neck.output)
classify = Dense(256, activation='relu')(classify)
classify = Dropout(0.5)(classify)
classify = Dense(1, activation='sigmoid')(classify)


# In[ ]:


Detector=Model(neck.input, classify)


# In[ ]:


for layers in neck.layers:
    layers.trainable=False


# In[ ]:


Detector.summary()


# In[ ]:


Detector.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


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
print("traindata = "+str(X_train.shape[0]))
print("testdata = "+str(X_test.shape[0]))


# In[ ]:


X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.


# In[ ]:


batch_size=1
num_batches=math.ceil(len(X_train)*1.0/batch_size)
for epoch in range(1):
    X_train,Y_train=shuffle(X_train,Y_train)
    print ("Epoch is: %d\n" % epoch)
    print ("Number of batches: %d\n" % int(num_batches))
    for batch in range(int(num_batches)):
        batch_train_X=X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
        batch_train_Y=Y_train[batch*batch_size:min((batch+1)*batch_size,len(Y_train))]
        loss=Detector.train_on_batch(batch_train_X,batch_train_Y)
#         l=Detector.predict(batch_train_X,verbose=0)
#         loss=binary_crossentropy(batch_train_Y,l)
#         print(loss)
        print ('epoch_num: %d batch_num: %d train loss: %f class-%f\n' % (epoch+1,batch+1,loss[0]*100,loss[1]))
Detector.save_weights("part1.h5")
print("PART1 - training completed for the classifier")


# In[ ]:


loss=0
num_batches=math.ceil(len(X_test)*1.0/batch_size)
for batch in range(int(num_batches)):
    batch_test_X=X_test[batch*batch_size:min((batch+1)*batch_size,len(X_test))]
    batch_test_Y=Y_test[batch*batch_size:min((batch+1)*batch_size,len(Y_test))]
    loss+=Detector.test_on_batch(batch_test_X,batch_test_Y)[0]
loss/=num_batches
print("Part1-Testloss="+str(loss*100)+"%")


# In[ ]:


FDetector=Model(neck.input,classify)


# In[ ]:


FDetector.summary()


# In[ ]:


FDetector.load_weights('part1.h5')


# In[ ]:


FDetector.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


batch_size=1
num_batches=math.ceil(len(X_train)*1.0/batch_size)
for epoch in range(1):
    X_train,Y_train=shuffle(X_train,Y_train)
    print ("Epoch is: %d\n" % epoch)
    print ("Number of batches: %d\n" % int(num_batches))
    for batch in range(int(num_batches)):
        batch_train_X=X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
        batch_train_Y=Y_train[batch*batch_size:min((batch+1)*batch_size,len(Y_train))]
        loss=FDetector.train_on_batch(batch_train_X,batch_train_Y)
#         l=Detector.predict(batch_train_X,verbose=0)
#         loss=binary_crossentropy(batch_train_Y,l)
#         print(loss)
        print ('epoch_num: %d batch_num: %d train loss: %f class-%f\n' % (epoch+1,batch+1,loss[0]*100,loss[1]))
FDetector.save_weights("detector.h5")
print("PART2 - training completed for the network")


# In[ ]:


loss=0
num_batches=math.ceil(len(X_test)*1.0/batch_size)
for batch in range(int(num_batches)):
    batch_test_X=X_test[batch*batch_size:min((batch+1)*batch_size,len(X_test))]
    batch_test_Y=Y_test[batch*batch_size:min((batch+1)*batch_size,len(Y_test))]
    loss+=FDetector.test_on_batch(batch_test_X,batch_test_Y)[0]
loss/=num_batches
print("Final Testloss="+str(loss*100)+"%")

