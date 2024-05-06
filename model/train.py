# Author: Sungmin Cho and Sujin Shin
# Adapted from the sign language recognition project described in the article:
# https://towardsdatascience.com/sign-language-recognition-with-advanced-computer-vision-7b74f20f3442
# Credit and acknowledgment to the original author for their valuable contributions.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# Step 1: Load and Prepare Data
def load_and_prepare_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df['label']
    y_test = test_df['label']
    del train_df['label']
    del test_df['label']

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    x_train = train_df.values
    x_test = test_df.values

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)

# Step 2: Data Augmentation
def create_data_generator():
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )
    return datagen

# Step 3: Define Model Architecture
def create_model():
    model = Sequential([
        Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding='same'),
        Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding='same'),
        Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding='same'),
        Flatten(),
        Dense(units=512, activation='relu'),
        Dropout(0.3),
        Dense(units=24, activation='softmax')
    ])
    return model

# Step 4: Train Model
def train_model(model, datagen, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=10, validation_data=(x_test, y_test))
    
    return history

# Step 5: Save Model
def save_model(model, filename):
    model.save(filename)

# Main Function
def main():
    train_path = '/Users/sungmincho/Desktop/ECE479/ece479github/InclusiveIoT/mnist/sign_mnist_train.csv'
    test_path = '/Users/sungmincho/Desktop/ECE479/ece479github/InclusiveIoT/mnist/sign_mnist_test.csv'
    
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data(train_path, test_path)
    datagen = create_data_generator()
    model = create_model()
    history = train_model(model, datagen, x_train, y_train, x_test, y_test)
    save_model(model, 'saved_model2.h5')

if __name__ == "__main__":
    main()

