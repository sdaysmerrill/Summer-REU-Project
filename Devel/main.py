#####################################################################
##                    Sarah Days-Merrill                           ##
##             CCT REU Louisiana State University                  ##
#####################################################################


import argparse
from data import *
from model import *
from train import *
#from test import testNet



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-config', help='config file to set.')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    
   
    if args.config == 'encdec.cfg':
        config = 'encdec.cfg'    #set config file
        
        
    else:
       config = None

    dmodel = Model()  #Model() is the class in data.py that deals with data processing
    call_model = dmodel.initNet(config)
    
    #data loading
    data_load = Model(config, opts = None)

    train = Train(dmodel)  #Train() is the class in train.py
    if args.mode == 'train':
        print("Begin training")
        train.trainNet()  #call the function train() in the class Train()
        
    elif args.mode == 'test':
        print("Start testing")
 #       testNet()  

    else:
        print("Please specify a mode")
        






   






