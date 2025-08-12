# Image-Processing-3
Image Classification

import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf
from pathlib import Path as path
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")

# Directory containing the dataset
data_dir = 'C:/Users/Asus/Downloads/Tensor/New folder/seg_train/seg_train'
save_dir = 'C:/Users/Asus/Downloads/Tensor/New folder/Predicted_Images'  # Directory to save correctly predicted images

# Define parameters for image size and batch size
img_height = 180
img_width = 180
batch_size = 32

# Use ImageDataGenerator to load images in batches and split dataset
datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to [0, 1]
    validation_split=0.2      # Reserve 20% of the data for validation
)

# Load training data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True,
    seed=None  # Set seed to None for better randomness
)


# Load validation data
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='sparse',
    subset='validation',
    shuffle=True,
    seed=None  # Set seed to None for better randomness
)


# Define class names
class_names = {
    0: 'building',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}


# Retrieve a truly random batch of images and labels
X_train_scaled, y_train = next(train_generator)  # Get the next batch of images and labels



# Randomly select an index from the batch
index_to_display = np.random.randint(len(X_train_scaled))
random_image = X_train_scaled[index_to_display]


# Display the random image
plt.imshow(random_image)
plt.title(f'Class: {class_names[int(y_train[index_to_display])]}')
plt.axis('off')
plt.show()


# Print the scaled pixel values of the image
print(f'Scaled pixel values of the image')
print(random_image)



num_classes = 6

# CNN model building
model = tf.keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),  # Convolutional layer
    layers.MaxPooling2D(2, 2),  # Pooling layer
    layers.Dropout(0.25),  # Dropout layer
      
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),  # Convolutional layer
    layers.MaxPooling2D(2, 2),  # Pooling layer
    layers.Dropout(0.25),  # Dropout layer

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),  # Convolutional layer
    layers.MaxPooling2D(2, 2),  # Pooling layer
    layers.Dropout(0.25),  # Dropout layer

    layers.Flatten(),  # Flatten layer
    layers.Dense(128, activation='relu'),  # Fully connected layer
    layers.Dropout(0.5),  # Dropout layer
    layers.Dense(num_classes, activation='softmax')  # Output layer
])



model.summary() # summary of the model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(
    train_generator,  # Training data generator
    epochs=20,  # Number of epochs to train the model
    validation_data=validation_generator,  # Validation data generator
    steps_per_epoch=train_generator.samples // batch_size,  # Number of batches per epoch
    validation_steps=validation_generator.samples // batch_size  # Number of batches in validation
)


# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# Make predictions on a random batch from the validation set
X_val_scaled, y_val = next(validation_generator)  # Get a batch of validation data

# Predict the classes for the batch
predictions = model.predict(X_val_scaled)

# Randomly select an index from the batch for visualization
index_to_display = np.random.randint(len(X_val_scaled))

# Get the predicted class for the selected sample
predicted_class = np.argmax(predictions[index_to_display])

# Get the true class for the selected sample
true_class = int(y_val[index_to_display])  # Convert to integer if needed

# Display the selected image with the predicted and true classes
plt.imshow(X_val_scaled[index_to_display])
plt.title(f'Predicted Class: {class_names[predicted_class]}, True Class: {class_names[true_class]}')
plt.axis('off')  # Hide the axes
plt.show()


# Print the result of the prediction
if predicted_class == true_class:
    print(f'Correct prediction! Predicted: {class_names[predicted_class]}, True: {class_names[true_class]}')
else:
    print(f'Incorrect prediction. Predicted: {class_names[predicted_class]}, True: {class_names[true_class]}')


#------------------------------------------------------RUN-------------------------------------------------------------


# final step to evaluate the model on the test data and save the predicted images along with their classified labels in the appropriate directories
test_dir = 'C:/Users/Asus/Downloads/Tensor/New folder/seg_test/'
save_dir = 'C:/Users/Asus/Downloads/Tensor/New folder/Predicted_Images'  # Directory to save correctly predicted images

# Create directories for each class inside the save_dir
for class_name in class_names.values():
    os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)

# Use ImageDataGenerator for the test set (without validation split)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='sparse',
    shuffle=False  # No shuffling so that results are consistent with file order
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Predict on the test set
test_generator.reset()  # Reset the generator to avoid any side effects
predictions = model.predict(test_generator)

# Iterate over each image in the test set
for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])  # Get predicted class index
    true_class = int(test_generator.labels[i])  # Get true class index

    # Get the file path of the current image
    img_path = test_generator.filepaths[i]
    img_name = os.path.basename(img_path)

    # Load the image (optional, for display or saving later)
    img = PIL.Image.open(img_path)

    # Print prediction result
    if predicted_class == true_class:
        print(f'Correctly predicted: {class_names[predicted_class]} for image {img_name}')
    else:
        print(f'Incorrectly predicted: {class_names[predicted_class]} (True: {class_names[true_class]}) for image {img_name}')

    # Save the image to the predicted class directory
    save_path = os.path.join(save_dir, class_names[predicted_class], img_name)
    img.save(save_path)

print(f'All predicted images saved to {save_dir}.')
    
