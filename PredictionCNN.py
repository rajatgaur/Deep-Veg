# load_model_sample.py
from keras.models import load_model
import tensorflow as tf
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from numpy import *








def create_data(path,img_size):

    tf.logging.set_verbosity(tf.logging.ERROR)
    X = []

    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, img_size)

    plt.imshow(img)
    plt.show()
    X = (np.array(img))/255




    return X





if __name__ == "__main__":
    



    filename = askopenfilename()

    img_path = filename

    X = create_data(img_path,(200,200))
    X = np.expand_dims(X, axis=0)
    #print(X.shape)

    # load model
    model = load_model("my_model1.h5")



    # check prediction

    pred = model.predict(X)
    pred = list(pred)
    #print(pred)
    pred = pred[0]
    print()
    print()
    print()
    if(pred[0]>pred[1]):
        print("Your Potato is Bad")
    else :
        print("Your Potato is Gooooood!!")
    print()
    print()
