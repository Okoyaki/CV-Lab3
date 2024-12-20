import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pathlib
import pickle

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout, Flatten, BatchNormalization,MaxPooling2D

def load_data(fl_imgs, fl_labels, width, height, channels):
    
    X, Y = [], []
    for flower_name, images in fl_imgs.items():
        for image in images:
            img = cv2.imread(str(image))
            if isinstance(img,type(None)): 
                continue
            elif ((img.shape[0] >= height) and  (img.shape[1] >=width)):
                resized_img = cv2.resize(img,(width,height))
                X.append(resized_img)
                Y.append(fl_labels[flower_name])
            else:
                continue
    return np.array(X), np.array(Y)

def train_and_save_model(X, Y, width, height, channels):
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(flower_labels_dict), activation='softmax') 
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('data/saved_model/flower_model.keras')
    history=model.fit(X_train, Y_train, epochs=20)
    with open('data/history_dict/hist', 'wb') as file:
        pickle.dump(history.history, file)
    print(model.evaluate(X_test,Y_test))

if __name__ == '__main__':
    
    data_dir = pathlib.Path('data/images/') 
    flower_images_dict = {
        'daisy': list(data_dir.glob('daisy/*')),
        'dandelion': list(data_dir.glob('dandelion/*')),
        'roses': list(data_dir.glob('roses/*')),
        'sunflowers': list(data_dir.glob('sunflowers/*')),
        'tulips': list(data_dir.glob('tulips/*'))
    }
    flower_labels_dict = {
        'daisy': 0,
        'dandelion': 1,
        'roses': 2,
        'sunflowers': 3,
        'tulips': 4
    }
    width=128
    height=128
    channels=3

    X, Y = load_data(flower_images_dict, flower_labels_dict, width, height, channels)
    train_and_save_model(X, Y, width, height, channels)

