# Extracting train,test,validation dataset
import zipfile 
f = zipfile.ZipFile('archive.zip','r') 
f.extractall()
f.close()

# Importing necessary libraries
import tensorflow as tf 
from tensorflow import keras 
from keras import Sequential 
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Generators 
train_ds = keras.utils.image_dataset_from_directory(
    directory='train', 
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)   #CNN Model expects all images to have sm dimensions
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory='val', 
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)   #CNN Model expects all images to have sm dimensions
) 

# Normalizing image array 
def process(image,label):
    image = tf.cast(image/255. , tf.float32)
    return image,label 

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# Creating CNN Model
model = Sequential() 

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds,epochs=20,validation_data=validation_ds)
