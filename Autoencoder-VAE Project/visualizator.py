from operator import ge
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

class Visualizator():

    def __init__(self) -> None:
        pass
    
    def differencePlots(self, originalImgs: np.ndarray= None, reconstrImgs: np.ndarray = None, color:bool = False, generative: bool = False):

        if(not generative):  
            rows = 2
            columns = 6
            ax = [] 
            fig = plt.figure(figsize=(20,20))
            for i in range(rows*columns):
                ax.append(fig.add_subplot(rows, columns, i+1))
                if(i <= 5):
                    ax[-1].set_title("original image: " + str(i))
                    plt.imshow(originalImgs[i], alpha=0.25)

                else:
                    ax[-1].set_title("reconstructed image: " + str(i-6))
                    plt.imshow(reconstrImgs[i-6], alpha=0.25)
        else:
            rows = 2
            columns = 8
            ax = [] 
            fig = plt.figure(figsize=(20,20))
            for i in range(rows*columns):
                ax.append(fig.add_subplot(rows, columns, i+1))
                plt.imshow(reconstrImgs[i], alpha=0.25)


        plt.show()
    def learningPlots(self, learningProgress):
        pass
