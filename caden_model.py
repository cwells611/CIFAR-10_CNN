# COMPETITION

#########################################
# DO NOT MODIFY THIS SECTION

# load the competition data
# The data is in the numpy array format:
#   competition_images: (100,32,32,3) contains 100 images
#   competition_labels: (100,1) contains class lables (0 to 9)
import numpy as np
competition_data = np.load('competition_data.npz') 
competition_images = competition_data['competition_images']
competition_labels = competition_data['competition_labels']


#########################################
# YOUR CODE/MODEL GOES HERE:
import tensorflow as tf
from tensorflow.keras import models, datasets
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#load CIFAR-10 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#noramlize training and testing
train_images = train_images / 255.0
test_images = test_images / 255.0

#create model 
model = models.Sequential()

"""
conv2d layers with 32 filters per layer
include batch normalization between layers to ensure data re-normalization 
dimensionality reduction via 2d max pooling 
dropout of 0.2 to reduce overfitting 
"""
input_shape = train_images[0].shape
print(input_shape)
model.add(Input(shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))

"""
conv2d layers with 64 filters per layer
include batch normalization between layers to ensure data re-normalization 
dimensionality reduction via 2d max pooling 
dropout of 0.2 to reduce overfitting 
"""
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))

"""
conv2d layers with 128 filters per layer
include batch normalization between layers to ensure data re-normalization 
dimensionality reduction via 2d max pooling 
dropout of 0.2 to reduce overfitting 
"""
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.2))

#flatten 2D output into 1D vector to pass into fully connected layer 
model.add(Flatten())

#fully connected layer with 128 neurons along with final normalization and droupout 
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

#output layer with 10 neurons, 1 for each class, and softmax for output of probabilities 
model.add(Dense(10, activation='softmax'))

model.summary()

#compile model
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

#augment the data to increase diversity of training set 
data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

#trainig variables
batch_size = 32
n_epochs = 35

#apply data augmentation to training images 
train_augmented = data_generator.flow(train_images, train_labels, batch_size=batch_size)

#train model
history = model.fit(train_augmented, batch_size=batch_size, epochs=n_epochs, validation_data=(test_images, test_labels))

#save model
model.save('my_model.keras')

#accuracy plot 
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()
plt.show()

#loss plot
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.show()

# load your model and/or trained weights
my_model = models.load_model('my_model.keras')

#evaluate using cifar-10 test set 
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')

# evaluate your model on the competition data
# make any adjustment to the data format as needed to run your model
# you must return accuracy of your model on the competition data 
competition_loss, competion_acc = my_model.evaluate(competition_images,  competition_labels)

# MUST PRINT OUT THE ACCURACY OF YOUR MODEL ON THE COMPETITION DATA
print('Accuracy:', competion_acc)