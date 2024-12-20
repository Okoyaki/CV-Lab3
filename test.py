import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pathlib
import pickle

from sklearn.model_selection import train_test_split
import tensorflow as tf

def evaluate_model(model, test_img):
    
    history = pickle.load(open('data/history_dict/hist', "rb"))
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['loss'], label = 'loss')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig('data/result/train.png')

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    labels = {
        0: 'daisy',
        1: 'dandelion',
        2: 'roses',
        3: 'sunflowers',
        4: 'tulips'
    }
    for ind, img in enumerate(test_imgs):
        plt.imshow(img)
        myimage = img.reshape(1, width, height, 3)
        print(model.predict(myimage))
        plt.title(labels[np.argmax(model.predict(myimage))])
        plt.savefig('data/result/test'+str(ind)+'.jpg')


if __name__ == '__main__':
    
    width=128
    height=128
    channels=3

    model = tf.keras.models.load_model('data/saved_model/flower_model.keras')
    test_img_dir = ['data/images/test/daisy.jpg', 'data/images/test/dandelion.jpg', 'data/images/test/sunflower.jpg']
    test_imgs = []
    for dir in test_img_dir:
        img = cv2.imread(dir)
        test_imgs.append(cv2.resize(img,(width,height)))
    evaluate_model(model, np.array(test_imgs))
