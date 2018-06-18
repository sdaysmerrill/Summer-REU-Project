from data import *
from models import *
from test import *
from train import *


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#%matplotlib inline

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

show_plot(plot_losses)
