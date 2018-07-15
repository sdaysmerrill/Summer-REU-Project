#####################################################################
##                    Sarah Days-Merrill                           ##
##             CCT REU Louisiana State University                  ##
#####################################################################


from model import EncDecRNN
from data import Model

import random
import time
import math
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

hidden_size = 500
n_layers = 2
dropout_p = 0.05
MAX_LENGTH = 10



class Train(Model):
        def __init__(self, dmodel):
 #               super(Model, self).__init__()
                self.encdecrnn = dmodel.model #EncDecRNN(Model.initNet, Model.initNet.self.dh, n_layers, dropout_p)
                print(self.encdecrnn)
        def trainNet(self):    #input_variable, target_variable, encdec, encdec_optimizer, criterion, max_length=MAX_LENGTH):

                ######## training RNN generator with early stopping ######### 
                if dmodel.debug:
                    print 'start network training ...'
                epoch = 0
                lr_divide = 0
                llr_divide= -1

                while True:
                    # training phase
                    epoch += 1
                    tic = time.time()
                    wcn, num_sent, train_logp = 0.0,0.0,0.0
                    while True:
                        # read data point
                        data = dmodel.reader.read(mode='train',batch=dmodel.batch)
                        if data==None:
                            break
                        # set regularization , once per ten times
                        reg = 0 if random.randint(0,9)==5 else dmodel.beta  #half the time reg = 0
                        # unfold data point
                        a,sv,s,v,words, _, _,cutoff_b,cutoff_f = data
                        print(a)  #[[7]]
                        print(sv) #[[82 101]]
                        print(s)  #[[24 31]]
                        print(v)  #[[1 1]]
                        print(words)  #[[1]
                                      #[ 5]
                                      #[ 2]
                                      #[ 4]
                                      #[ 33]
                                      #[105]
                                      #[ 11]
                                      #[ 3]
                                      #[ 22]
                                      #[ 21]
                                      #[ 13]
                                      #[ 1]]
                        # train net using current example 
                        train_logp += dmodel.model.train( a,sv,s,v,words,
                                cutoff_f, cutoff_b, dmodel.lr, reg)   #too many arguments error
                        # count words and sents 
                        wcn += np.sum(cutoff_f-1)
                        num_sent+=cutoff_b
                        # log message 
                        if dmodel.debug and num_sent%100==0:
                            print 'Finishing %8d sent in epoch %3d\r' % \
                                    (num_sent,epoch),
                            sys.stdout.flush()
                    # log message
                    sec = (time.time()-tic)/60.0
                    if dmodel.debug:
                        print 'Epoch %3d, Alpha %.6f, TRAIN entropy:%.2f, Time:%.2f mins,' %\
                                (epoch, dmodel.lr, -train_logp/log10(2)/wcn, sec),
                        sys.stdout.flush()

                    # validation phase
                    dmodel.valid_logp, wcn = 0.0,0.0
                    while True:
                        # read data point
                        data = dmodel.reader.read(mode='valid',batch=dmodel.batch)
                        if data==None:
                            break
                        # unfold data point
                        a,sv,s,v,words, _, _,cutoff_b,cutoff_f = data
                        # validating
                        dmodel.valid_logp += dmodel.model.test( a,sv,s,v,words,
                                cutoff_f, cutoff_b )
                        wcn += np.sum(cutoff_f-1)
                    # log message
                    if dmodel.debug:
                        print 'VALID entropy:%.2f'%-(dmodel.valid_logp/log10(2)/wcn)

                    # decide to throw/keep weights
                    if dmodel.valid_logp < dmodel.llogp:
                        dmodel.updateTheanoParams()
                    else:
                        dmodel.updateNumpyParams()
                    dmodel.saveNet()
                    # learning rate decay
                    if lr_divide>=dmodel.lr_divide:
                        dmodel.lr *= dmodel.lr_decay
                    # early stopping
                    if dmodel.valid_logp*dmodel.min_impr<dmodel.llogp:
                        if lr_divide<dmodel.lr_divide:
                            dmodel.lr *= dmodel.lr_decay
                            lr_divide += 1
                        else:
                            dmodel.saveNet()
                            print 'Training completed.'
                            break
                    # set last epoch objective value
                    dmodel.llogp = dmodel.valid_logp


 #       def as_minutes(s):
#            m = math.floor(s / 60)
#            s -= m * 60
#            return '%dm %ds' % (m, s)

#        def time_since(since, percent):
#            now = time.time()
#            s = now - since
#            es = s / (percent)
#            rs = es - s
#            return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


                # Configuring training
#        n_epochs = 500    #this is supposed to be 50000
#        plot_every = 250  #200
#        print_every = 100 #1000

                # Keep track of time elapsed and running averages
#        start = time.time()
#        plot_losses = []
#        print_loss_total = 0 # Reset every print_every
#        plot_loss_total = 0 # Reset every plot_every


                #input_varaible = variables_from_pair.input_variable
                #target_variable = variables_from_pair.target_variable

                # Begin!
 #       for epoch in range(1, n_epochs + 1):
                    
                    # Get training data for this cycle
         #   training_pair = variables_from_pair(random.choice(pairs))
        #    input_variable = training_pair[0]
        #    target_variable = training_pair[1]

                    # Run the train function
                
 #           loss = trainNet()  #input_variable, target_variable, encdec, encdec_optimizer, criterion)

                    # Keep track of loss
 #           print_loss_total += loss
#            plot_loss_total += loss

#            if epoch == 0: continue

#            if epoch % print_every == 0:
#                print_loss_avg = print_loss_total / print_every
#                print_loss_total = 0
#                print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch *1.0 / n_epochs * 1.0), epoch, epoch*1.0 / n_epochs*1.0 * 100, print_loss_avg)
#                print(print_summary)

#            if epoch % plot_every == 0:
#                plot_loss_avg = plot_loss_total / plot_every
#                plot_losses.append(plot_loss_avg)
#                plot_loss_total = 0
 

        
#        plt.figure()
#        plt.plot(plot_losses)

dmodel = Model()
call_model = dmodel.initNet('encdec.cfg')
training = Train(dmodel)



