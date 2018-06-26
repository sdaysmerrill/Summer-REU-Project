import argparse

from net import trainNet  #when I import net.py it automatically runs through all the code

from data import *

USE_CUDA = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    args = parser.parse_args()

    
    if args.mode == 'train':
        print("Begin training")
        #need to define the variables inside the function
        #these variables are in data.py - how do we get them here??
        trainNet(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH())

    elif args.mode == 'test':
        print("Start testing")
 #       testNet()

    else:
        print("Please specify a mode")
