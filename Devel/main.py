import argparse
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
 #       trainNet(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10)
       # os.system('python net.py')
        import train
        
    elif args.mode == 'test':
        print("Start testing")
        import test
 #       testNet()

    else:
        print("Please specify a mode")
