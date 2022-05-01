import cv2
import os
import pandas as pd
import numpy as np

def load_input(label: str, step: str, n_qbits: int = 10):
    """
    Loads the dataset for 'label' and converts all images to input vectors and returns it.

    @param label: The label to load the dataset for. [airplanes, ships]
    @param step: The step to load the dataset for. [train, test]
    @param n_qbits: The number of qubits for resizing image. Default: 10
    @return: The dataset as a list of input vectors.
    """
    _dir = f'Dataset/imgs/{step}/{label}/'
    files = os.listdir(_dir)
    inputs = []

    for file in files:
        img = cv2.imread(_dir + file)
        # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize
        size = int(2**(n_qbits/2))
        img = cv2.resize(img, (size,size))
        img = img.flatten()
        # normalize
        img = (img*np.pi/2)/255
        inputs.append(img)
    
    return inputs

def train_df(n=32):
    return pd.read_pickle(f'Dataset/train_{n}.pkl')

def test_df(n=32):
    return pd.read_pickle(f'Dataset/test_{n}.pkl')
