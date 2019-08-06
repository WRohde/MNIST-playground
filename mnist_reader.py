import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

def load_mnist_df(filepath='data/',test=False):
    mndata = MNIST(filepath)
    if(test == False):
        images, labels = mndata.load_training()
    else:
        images,labels = mndata.load_testing()

    df = pd.DataFrame(images,dtype='int8')
    df['label'] = labels
    return df

def show_mnist_image(df, index = None):
    """
    shows an image froma pandas dataframe of mnist images (784 columns
    of image data plus a label column)
    """
    if(index == None):
        index = np.random.randint(0,len(df))
    
    try:
        image_df = df.drop("label",axis=1)
        label_exists = True
    except:
        image_df = df
        label_exists = False

    image_array = np.reshape(image_df.loc[index,:].values, (28,28))
    
    plt.imshow(image_array)
    if label_exists: plt.title(str(df.label.loc[index]),)
    plt.show() 
