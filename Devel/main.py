#####################################################################
##                    Sarah Days-Merrill                           ##
##             CCT REU Louisiana State University                  ##
#####################################################################


import argparse
from data import *
from model import *
#from train import trainRNN
#from test import testNet

MAX_LENGTH = 10

#teacher_forcing_ratio = 0.5
#clip = 5.0


hidden_size = 500
n_layers = 1
dropout_p = 0.05
learning_rate = 0.0001

# Initialize model
encdecrnn = EncDecRNN() #input_size, hidden_size, n_layers, dropout_p = dropout_p)  #add parameters




# Move models to GPU
#if USE_CUDA:
#    encoder.cuda()
#    decoder.cuda()

# Initialize optimizers and criterion



#encdec_optimizer = optim.SGD(encdec.parameters(), lr=learning_rate)
#error: has empty parameter list
#criterion = nn.NLLLoss()




#this is the end of main file
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


    
    if args.cuda == 'yes':
        USE_CUDA = True
        encdecrnn.cuda()

    else:
        USE_CUDA = False

    if args.config == 'encdec.cfg':
        config = 'encdec.cfg'
        
        
    else:
       config = None

    #data loading
    data_load = Model(config, opts = None)
    
    if args.mode == 'train':
        print("Begin training")
 #       training = trainRNN()
        
    elif args.mode == 'test':
        print("Start testing")
 #       testNet()  

    else:
        print("Please specify a mode")
        






   






