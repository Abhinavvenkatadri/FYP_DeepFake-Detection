import os
import io
import cv2
import json
import time
import random
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from focal_loss import BinaryFocalLoss
from sklearn.model_selection import train_test_split

SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
tf.random.set_seed(SEED)

#!pip install keras==2.6.0
#!rm -rf DIR_NAME



######################   CONFIG   ###################### 

# Opening JSON file
f = open('CONFIG.json',)
CONFIG = json.load(f)
print(CONFIG, "\n")

FILE_NAME = CONFIG["FILE_NAME"]

#os.mkdir(FILE_NAME)
print("File Created : ", FILE_NAME, "\n")
out_file = open(FILE_NAME + "/" + "CONFIG.json", "w")  
json.dump(CONFIG, out_file, indent = 8)
out_file.close()

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



'''
paths = []
labels  = []
for video in os.listdir('faceforensics/manipulated_sequences/Deepfakes/c23/videos'):
    vid_file = os.path.join('faceforensics/manipulated_sequences/Deepfakes/c23/videos', video)
    paths.append(vid_file)
    labels.append([1])
for video in os.listdir('faceforensics/original_sequences/youtube/c23/videos'):
    vid_file = os.path.join('faceforensics/original_sequences/youtube/c23/videos', video)
    paths.append(vid_file)
    labels.append([0])

'''


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

def normalize_image(matrix):
    return (matrix - np.mean(matrix))/np.std(matrix)

'''
def video_generator(data_paths):
    
    for i in range(0, (len(data_paths)//BATCH_SIZE)*BATCH_SIZE, BATCH_SIZE):
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
        batch_vids = np.asarray(batch_vids)/255
        batch_labels = np.asarray(batch_labels)
        #print("\n")
        #[print(x[0], end = " ") for x in batch_labels]
        #print("\n")
        #[print(x[0], end = " ") for x in model.predict(batch_vids)]
        #print("\n")
        yield normalize_image(batch_vids), batch_labels
'''

'''
next_batch = None
is_batch_trainied = True

def prepare_next_batch(data_paths):
    global next_batch
    
    batch_vids = []
    batch_labels = []

    for path, start, end, label in data_paths:
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
    batch_vids = normalize_image(np.asarray(batch_vids)/255)
    batch_labels = np.asarray(batch_labels)
    while True:
        if is_batch_trainied:
            next_batch = (batch_vids, batch_labels)
            break
            
            
def video_generator(data_paths):
    global next_batch
    global is_batch_trainied
    
    t = threading.Thread(target=prepare_next_batch, args=(data_paths[:BATCH_SIZE],))
    t.start()
    for i in range(1, (len(data_paths)//BATCH_SIZE)*BATCH_SIZE, BATCH_SIZE):
        is_batch_trainied = True
        t.join()
        batch_vids, batch_labels = next_batch
        is_batch_trainied = False
        t = threading.Thread(target=prepare_next_batch, args=(data_paths[i:i+BATCH_SIZE],))
        t.start()
        yield batch_vids, batch_labels
'''

'''
next_batch = None
is_batch_loaded = True
batch_vids = [None]*BATCH_SIZE
batch_labels = [None]*BATCH_SIZE


def load_video(j, data_path):
    global batch_vids
    global batch_labels
    
    path, start, end, label = data_path
    
    cap = cv2.VideoCapture(path)
    frameCount, frameWidth, frameHeight = INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]
    vid = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('float'))
    fc = 0
    ret = True
    for k in range(end):
        ret, image = cap.read()
        if k >= start:
            vid[fc] = cv2.resize(image, (frameWidth, frameHeight))
            vid = vid.astype(float)
            fc += 1
    
    batch_vids[j] = vid
    batch_labels[j] = label
    
def prepare_next_batch(data_paths):
    global next_batch
    global batch_vids
    global batch_labels
    global thread_list
    
    #batch_vids = [None]*BATCH_SIZE
    #batch_labels = [None]*BATCH_SIZE
    thread_list = []
    
    for j, data_path in enumerate(data_paths):        
        thread = threading.Thread(target=load_video, args=(j, data_path))
        thread_list.append(thread)
        thread.start()
    
    for thread in thread_list:
        thread.join()
    
    batch_vids = normalize_image(np.asarray(batch_vids)/255)
    batch_labels = np.asarray(batch_labels)
    while True:
        if is_batch_loaded:
            next_batch = (batch_vids, batch_labels)
            break
            
            
def video_generator(data_paths):
    global next_batch
    global batch_vids
    global batch_labels
    global thread_list
    
    t = threading.Thread(target=prepare_next_batch, args=(data_paths[:BATCH_SIZE],))
    t.start()
    for i in range(1, (len(data_paths)//BATCH_SIZE)*BATCH_SIZE, BATCH_SIZE):
        is_batch_loaded = True
        t.join()
        batch_vids, batch_labels = next_batch
        is_batch_loaded = False
        t = threading.Thread(target=prepare_next_batch, args=(data_paths[i:i+BATCH_SIZE],))
        t.start()
        yield batch_vids, batch_labels     
'''

#BATCH_SIZE = 32
next_batch = None
#is_batch_loaded = True
batch_vids = [None]*BATCH_SIZE
batch_labels = [None]*BATCH_SIZE


def load_video(j, data_path):
    global batch_vids
    global batch_labels
    
    path, start, end, label = data_path
    
    cap = cv2.VideoCapture(path)
    frameCount, frameWidth, frameHeight = INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]
    vid = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('float'))
    fc = 0
    ret = True
    for k in range(end):
        ret, image = cap.read()
        if k >= start:
            vid[fc] = cv2.resize(image, (frameWidth, frameHeight))
            vid = vid.astype(float)
            fc += 1
    
    batch_vids[j] = vid
    batch_labels[j] = label

    
def prepare_next_batch(data_paths):
    global next_batch
    global batch_vids
    global batch_labels
    global thread_list
    
    thread_list = []
    for j, data_path in enumerate(data_paths):        
        thread = threading.Thread(target=load_video, args=(j, data_path))
        thread_list.append(thread)
        thread.start()
        #print("Started : ", j)
    for thread in thread_list[::-1]:
        thread.join()

    batch_vids = (2*(np.asarray(batch_vids)/255) - 1)    
    batch_labels = np.asarray(batch_labels)

    next_batch = (batch_vids, batch_labels)
    
            
def video_generator(data_paths):
    global next_batch
    global batch_vids
    global batch_labels
    global thread_list
    
    t = threading.Thread(target=prepare_next_batch, args=(data_paths[:BATCH_SIZE],))
    t.start()
    for i in range(1, (len(data_paths)//BATCH_SIZE)*BATCH_SIZE, BATCH_SIZE):
        t.join()
        batch_vids, batch_labels = next_batch
        t = threading.Thread(target=prepare_next_batch, args=(data_paths[i:i+BATCH_SIZE],))
        t.start()
        yield batch_vids, batch_labels



train_paths, test_paths, _, __ = train_test_split(data_paths, data_paths, test_size=0.2, random_state=42)



train_paths, validation_paths, _, __ = train_test_split(train_paths, train_paths, test_size=0.25, random_state=42)



data = {"train_paths":train_paths, "validation_paths":validation_paths, "test_paths":test_paths}



out_file = open(FILE_NAME + "/" + FILE_NAME + "_data_train_validation_test_split.json", "w")  
json.dump(data, out_file, indent = 8)
out_file.close()

print("Total Training Samples    :   ", len(train_paths))
print("Total Validation Samples  :   ", len(validation_paths))
print("Total Testing Samples     :   ", len(test_paths))



trainloader = tf.data.Dataset.from_generator(
    lambda: video_generator(train_paths), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]), (BATCH_SIZE, 1))
)

validloader = tf.data.Dataset.from_generator(
    lambda: video_generator(validation_paths), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]), (BATCH_SIZE, 1))
)


testloader = tf.data.Dataset.from_generator(
    lambda: video_generator(test_paths), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]), (BATCH_SIZE, 1))
)

print(trainloader)



class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches



class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens



def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    key_dim=KEY_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                layers.Dense(units=embed_dim, activation=tf.nn.gelu),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)
    outputs = layers.Dense(units=512, activation="relu")(representation)
    outputs = layers.Dense(units=128, activation="relu")(representation)
    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="sigmoid")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



def get_model():
    
    model = create_vivit_classifier(
            tubelet_embedder = TubeletEmbedding(
                embed_dim = PROJECTION_DIM,
                patch_size = PATCH_SIZE
            ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=BinaryFocalLoss(gamma=1),
        metrics=["accuracy"],
    )


    return model

#model = get_model()
model = tf.keras.models.load_model("model_1_bin_focal_gamma_1_threading_E_0_10/model/saved_model")
model.summary(line_length=125)



checkpoint_filepath = FILE_NAME + "/" + FILE_NAME + '/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_freq = "epoch")
'''
class NBatchLogger(tf.keras.callbacks.Callback):
    def __init__(self,display=5):
        self.seen = 0
        self.display = display

    def on_batch_end(self,batch,logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            print(self.__dict__.items())
'''

#out_batch = NBatchLogger(display=100)
#history = model.fit(trainloader, validation_data = validloader, epochs = EPOCHS, verbose = 1, use_multiprocessing = True, callbacks=[model_checkpoint_callback])
#history = model.fit(trainloader, validation_data = validloader, epochs = EPOCHS, verbose = 1, use_multiprocessing = True)

#out_file = open(FILE_NAME + "/" + FILE_NAME + "_history.json", "w")
#json.dump(history.history, out_file, indent = 8)
#out_file.close()

#model.save(FILE_NAME + "/model/saved_model")

model.evaluate(testloader)


