import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

def load_mnist_df(filepath='data/'):
    mndata = MNIST(filepath)
    images, labels = mndata.load_training()

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
    
    image_df = df.drop("label",axis=1)
    image_array = np.reshape(image_df.loc[index,:].values, (28,28))
    
    plt.imshow(image_array)
    plt.title(str(df.label.loc[index]),)
    plt.show()
