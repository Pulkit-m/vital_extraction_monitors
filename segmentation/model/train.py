import tensorflow as tf
import cv2 
from copy import deepcopy 
from tensorflow import keras
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt

# Create ImageDataGenerator objects
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Assign the image directories to them
train_data_generator = train_datagen.flow_from_directory(
    "./dataset/train",
    target_size=(512,512)
)

test_data_generator = train_datagen.flow_from_directory(
    "./dataset/test",
    target_size=(512,512)
)


resnet_50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
resnet_50.trainable=False

inputs = keras.Input(shape=(512,512,3))

x = resnet_50(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(64)(x)
outputs = keras.layers.Dense(4, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="my_model")
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
print(model.summary())


history = model.fit(train_data_generator, validation_data=test_data_generator, epochs=10)
model.save("./assets/segmentation_model.h5")


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
figure = plt.figure(figsize = (20,10))
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./assets/training.png')


def show_predictions(dataset, y_dataset, num, model):
    image = dataset[num]
    image = np.reshape(image, (1, 512, 512, 1))
    mask = y_dataset[num]
    pred_mask = model.predict(image)
    pred_mask = np.reshape(pred_mask, (512, 512, 1))
    display([image, mask, pred_mask])
    return image, mask, pred_mask

segmentation_model = keras.models.load_model("/content/drive/MyDrive/InterIIT-PS2-MID-PREP/segmentation_new_on_grayscale.h5")
image = cv2.imread("/content/drive/MyDrive/InterIIT-PS2-MID-PREP/DATA/Unlabelled_Dataset_extracted/aastha_icu_mon--5_2023_1_2_9_0_0.jpeg")

def segmentation(segmentation_model, image):
  SIZE_X = 512 #Resize images (height  = X, width = Y)
  SIZE_Y = 512
  img_og = deepcopy(image)
  image = cv2.resize(image, (SIZE_Y, SIZE_X))
  # image = np.reshape(image, (1, 256, 256, 3))
  image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_grayscale = np.reshape(image_grayscale, (1, 512, 512, 1))  
  pred_mask = segmentation_model.predict(image_grayscale)
  pred_mask = np.reshape(pred_mask, (512, 512, 1))
  pred_mask = cv2.resize(pred_mask, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
  return pred_mask, img_og


pred_mask, image = segmentation(segmentation_model, image)

