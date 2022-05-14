#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import io
import cv2
import json
import time
import math
import random
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from focal_loss import BinaryFocalLoss
from sklearn.model_selection import train_test_split

SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
tf.random.set_seed(SEED)


# In[8]:


get_ipython().system('pip install tensorflow==2.6.0')


# In[5]:


get_ipython().system('pip install keras==2.6.0')


# In[2]:




######################   CONFIG   ###################### 

# Opening JSON file
f = open('CONFIG.json',)
CONFIG = json.load(f)
print(CONFIG, "\n")

FILE_NAME = CONFIG["FILE_NAME"]

#os.mkdir(FILE_NAME)
#print("File Created : ", FILE_NAME, "\n")
#out_file = open(FILE_NAME + "/" + "CONFIG.json", "w")  
#json.dump(CONFIG, out_file, indent = 8)
#out_file.close()

# DATA
#AUTO = tf.data.AUTOTUNE
#INPUT_SHAPE = (8, 192, 192, 3)
INPUT_SHAPE = tuple(CONFIG["INPUT_SHAPE"])
NUM_CLASSES = CONFIG["NUM_CLASSES"]

# OPTIMIZER
LEARNING_RATE = CONFIG["LEARNING_RATE"]
#WEIGHT_DECAY = 1e-4

# TRAINING
EPOCHS = CONFIG["EPOCHS"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]

# TUBELET EMBEDDING
#PATCH_SIZE = (8, 8, 8)
PATCH_SIZE = tuple(CONFIG["PATCH_SIZE"])
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2
PROJECTION_DIM = CONFIG["PROJECTION_DIM"]

# ViViT ARCHITECTURE
LAYER_NORM_EPS = CONFIG["LAYER_NORM_EPS"]
NUM_HEADS = CONFIG["NUM_HEADS"]
KEY_DIM = CONFIG["KEY_DIM"]
NUM_LAYERS = CONFIG["NUM_LAYERS"]


# In[15]:


def get_cropped_section(n,i):
    x = (i%6)*32
    y = (i//6)*32
    return x, x+32, y, y+32, batch_vids[0][n][x:x+32, y:y+32]


# In[16]:


def get_sample(i):
    BATCH_SIZE = 1
    batch_vids = []
    batch_labels = []

    for path, start, end, label in data_paths[i:i+BATCH_SIZE]:
        batch_labels.append(label)

        cap = cv2.VideoCapture(path)
        frameCount, frameWidth, frameHeight = INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]
        vid = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('float'))
        fc = 0
        ret = True
        for i in range(end):
            ret, image = cap.read()
            if i >= start:
                vid[fc] = cv2.resize(image, (frameWidth, frameHeight))
                vid = vid.astype(float)
                fc += 1
        batch_vids.append(vid)
    batch_vids = (2*(np.asarray(batch_vids)/255) - 1)    
    batch_labels = np.asarray(batch_labels)
    return batch_vids, batch_labels


# In[4]:


model = tf.keras.models.load_model("model_1_bin_focal_gamma_1_threading_E_10_15/model/saved_model")


# In[5]:


filters = model.layers[1].get_weights()
filters = np.asarray(filters)
print(filters[0].shape)
biases = filters[1]
filters = filters[0]
filters.shape


# In[6]:


mean_filters = np.mean(filters, axis = 0)


# In[7]:


mean_filters = mean_filters.reshape((512, 32, 32, 3))


# In[8]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[13]:


convolution = model.layers[1].output
attention = model.layers[-8].output


# In[ ]:





# In[ ]:





# In[428]:


filter_no = []
for opt in conv_opt[0]:
    #filter_no.append(np.argmin(np.abs(opt)))
    filter_no.append(np.argmax(np.abs(opt)))
filter_no


# In[557]:


#for i in range(512):
img = filters[7]
print(img.shape)
img = img.reshape((512, 32, 32, 3))[2]
min_ = np.min(img)
img = img - min_
max_ = np.max(img)
img = img/max_
print(img.shape)
img = img*(batch_vids[0][0][32:64, 32:64])
plt.imshow(img)


# In[10]:





# In[556]:





# In[14]:


new_model = Model(inputs=model.input, outputs=[convolution, attention, model.output])


# In[ ]:





# In[17]:


paths = []
labels  = []
for video in os.listdir('/raid/Data/Sayali/FF_Video_Fake'):
    vid_file = os.path.join('/raid/Data/Sayali/FF_Video_Fake', video, 'project.avi')
    paths.append(vid_file)
    labels.append([1])
for video in os.listdir('/raid/Data/Sayali/FF_Video_Real'):
    vid_file = os.path.join('/raid/Data/Sayali/FF_Video_Real', video, 'project.avi')
    paths.append(vid_file)
    labels.append([0])


# In[18]:


f = open('frames_count_new.json',"r")
frames_count = json.load(f)
data_paths = []
for (path, max_frames), label in zip(frames_count.items(), labels):
    for i in range(max_frames//INPUT_SHAPE[0]):
        dt = [path, i*INPUT_SHAPE[0], (i+1)*INPUT_SHAPE[0], label]
        data_paths.append(dt)
#data_paths = data_paths[:(len(data_paths)//BATCH_SIZE)*BATCH_SIZE]
#data_paths = random.sample(data_paths, 10000)
print("Total Available Samples   :   ", len(data_paths))
#data_paths = data_paths[:5000]


# In[ ]:





# In[ ]:





# In[988]:


batch_vids, batch_labels = get_sample(4001)


# In[989]:


for n,frame_filters in enumerate(filters):
    if n!= 0: continue
    filter_list = frame_filters.reshape((512, 32, 32, 3))
    image = np.zeros((192,192,3))
    for i,filter_idx in enumerate(filter_no):
        filter_ = filter_list[filter_idx]
        #min_ = np.min(filter_)
        #filter_ = filter_ - min_
        #filter_ = filter_/np.max(filter_)
        x,w,y,h, img = get_cropped_section(n,i)
        #img = img*(1 + 0*filter_)
        
        #min_ = np.min(img)
        #img = img - min_
        #img = img/np.max(np.abs(img))
        #img = np.mean(img, axis = -1)
        image[x:w,y:h] = img[:,:,::-1]
        
    plt.imshow(image)
    break


# In[990]:


outputs = new_model.predict([batch_vids])
conv_opt, att_opt, preds = outputs


# In[991]:


att = np.asarray(att_opt[0])
attention = np.matmul(att, att.T)


# In[992]:


import matplotlib.pyplot as plt
import numpy as np
print(batch_labels[0][0], preds[0][0])
plt.imshow(attention, cmap='hot', interpolation='nearest')
plt.show()


# In[993]:


for i in range(len(attention)):
    attention[i][i] = 0
tubelet_att = np.sum(attention, axis = 0)
tubelet_att = tubelet_att - np.min(tubelet_att) 
tubelet_att = tubelet_att/np.max(tubelet_att)


# In[994]:


plt.imshow(attention, cmap='hot', interpolation='nearest')
plt.show()


# In[995]:


image = np.zeros((192,192,3))
att_img = np.zeros((192,192,3))
for i,att in enumerate(tubelet_att):
    n = 0
    x,w,y,h, img = get_cropped_section(n,i)
    img = 255*(img + 1)/2
    img = img[:,:,::-1]*att
    image[x:w,y:h] = img
    att_img[x:w,y:h] = att
image = image.astype(int)


# In[996]:


batch_image = (255*(batch_vids[0][0]+1)/2).astype(int)[:,:,::-1]
plt.imshow(batch_image)


# In[997]:


plt.imshow(image)


# In[998]:


plt.imshow(att_img)


# In[999]:


blur_radius = 75
gaussian = cv2.GaussianBlur(att_img, (blur_radius, blur_radius), 0)
gaussian = mix_max_scalar(gaussian)
#gaussian = sigmoid(7*(gaussian-0.5))
#gaussian = (1.2*gaussian)**1
#gaussian[gaussian<0.5] = 0.3
#gaussian = np.maximum(gaussian, 0.7)


# In[1000]:


blur_radius = 75
atts = att_img.copy()
atts[atts<0.5] = 0.2
gaussian = cv2.GaussianBlur(atts, (blur_radius, blur_radius), 0)
gaussian = mix_max_scalar(gaussian)


# In[1001]:


plt.imshow(gaussian)


# In[1002]:


plt.imshow(batch_image*np.asarray(gaussian)/255)


# In[1003]:


def mix_max_scalar(x):
    min_ = np.min(x)
    x = x - min_
    return x/np.max(x)


# In[ ]:





# In[ ]:




