#####################################################################
##                    Sarah Days-Merrill                           ##
##             CCT REU Louisiana State University                  ##
#####################################################################

from model import embedding
from model import DecoderRNN
import data

import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#these are from the tutorial - going to be removed later
#hidden_size = 500
#n_layers = 2
#dropout_p = 0.05
#MAX_LENGTH = 10

USE_CUDA = False

def trainNet(NetModel):    #input_variable, target_variable, encdec, encdec_optimizer, criterion, max_length=MAX_LENGTH):
                ######## training RNN generator with early stopping ######### 
        if NetModel.debug:
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
                        data = NetModel.reader.read(mode='train',batch=NetModel.batch)
                        if data==None:
                                break
                        # set regularization , once per ten times
                        reg = 0 if random.randint(0,9)==5 else NetModel.beta  #half the time reg = 0
                        # unfold data point
                        a,sv,s,v,words, _, _,cutoff_b,cutoff_f = data

 #                       tensor_a = torch.from_numpy(a)
#                        tensor_sv = torch.from_numpy(sv)
#                        tensor_s = torch.from_numpy(s)
#                        tensor_v = torch.from_numpy(v)
#                        tensor_words = torch.from_numpy(words)

                        
 #                       print("(debug) number of rows in tensor_a = ", tensor_a.size().size[0]) 
 #                       print(a)  #[[7]]
#                        print(sv) #[[82 101]]
#                        print(s)  #[[24 31]]
#                        print(v)  #[[1 1]]
#                        print(words)          #[[1]
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
 #                       train_logp += NetModel.model.train( a,sv,s,v,words,
#                                cutoff_f, cutoff_b, NetModel.lr, reg)   #too many arguments error
                        # count words and sents 
#                        wcn += np.sum(cutoff_f-1)
#                        num_sent+=cutoff_b



#this is our training
                # Zero gradients of both optimizers
 #                       encoder_optimizer = optim.SGD(NetModel.emodel.parameters(), lr = NetModel.lr)
#                        encoder_optimizer.zero_grad()
                        decoder_optimizer = optim.SGD(NetModel.dmodel.parameters(), lr = NetModel.lr)
                        decoder_optimizer.zero_grad()
                        loss = 0 # Added onto for each word

                    # Get size of input and target sentences - figure out how to get these
                        input_length = len(words)  #this just holds the string data/original/...
#                        print(tensor_words)
#                        print(input_length)
 #                       target_length = target_variable.size()[0]

                    # Run words through encoder
#                        encoder = EncoderRNN()
#                        encoder_hidden = NetModel.emodel.init_hidden()
                        #figure out why this is not recursive
                        
                        a_emb, sv_emb, words_emb = embedding(NetModel, a, s, v, words)

 #                       for ei in range(input_length):
#                                encoder_output, encoder_hidden = NetModel.emodel(tensor_words, encoder_hidden)
                                


                        SOS_token = 0
                    # Prepare input and output variables
                        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
                        decoder_context = Variable(torch.zeros(1, NetModel.dmodel.hidden_size))
#                        decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
                        if USE_CUDA:
                                decoder_input = decoder_input.cuda()
                                decoder_context = decoder_context.cuda()

                    # Choose whether to use teacher forcing
 #                       use_teacher_forcing = random.random() < teacher_forcing_ratio
#                        if use_teacher_forcing:
                        
                        # Teacher forcing: Use the ground-truth target as the next input
#                                for di in range(target_length):
#                                    decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
#                                    loss += criterion(decoder_output[0], target_variable[di])
#                                    decoder_input = target_variable[di] # Next target is next input

#                        else:
                        # Without teacher forcing: use network's own prediction as the next input

                        #do not use teacher forcing
                        for di in range(input_length):
                                decoder_output, decoder_context, decoder_hidden, decoder_attention = NetModel.dmodel(a_emb, sv_emb, words_emb)
                                loss += criterion(decoder_output[0], tensor_words[di])
                            
                            # Get most likely word index (highest value) from output
                                topv, topi = decoder_output.data.topk(1)
                                ni = topi[0][0]
                            
                                decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
                                if USE_CUDA: decoder_input = decoder_input.cuda()

                            # Stop at end of sentence (not necessary when using known targets)
                                if ni == EOS_token: break

                    # Backpropagation
                        loss.backward()
                        torch.nn.utils.clip_grad_norm(NetModel.emodel.parameters(), clip)
 #                      torch.nn.utils.clip_grad_norm(NetModel.dmodel.parameters(), clip)
#                        encoder_optimizer.step()
                        decoder_optimizer.step()
                    
                 #   return loss.data[0] / target_length




                        # log message 
                        if NetModel.debug and num_sent%100==0:
                                print 'Finishing %8d sent in epoch %3d\r' % \
                                        (num_sent,epoch),
                                sys.stdout.flush()
                    # log message
                sec = (time.time()-tic)/60.0
                if NetModel.debug:
                        print 'Epoch %3d, Alpha %.6f, TRAIN entropy:%.2f, Time:%.2f mins,' %\
                                (epoch, NetModel.lr, -train_logp/log10(2)/wcn, sec),
                sys.stdout.flush()

                    # validation phase
#                NetModel.valid_logp, wcn = 0.0,0.0
#                while True:
                        # read data point
#                        data = NetModel.reader.read(mode='valid',batch=NetModel.batch)
#                        if data==None:
#                                break
                        # unfold data point
#                        a,sv,s,v,words, _, _,cutoff_b,cutoff_f = data
                        # validating
 #               NetModel.valid_logp += NetModel.model.test( a,sv,s,v,words,
#                        cutoff_f, cutoff_b )
#                wcn += np.sum(cutoff_f-1)
                    # log message
#                if NetModel.debug:
#                        print 'VALID entropy:%.2f'%-(NetModel.valid_logp/log10(2)/wcn)

                    # decide to throw/keep weights
#                if NetModel.valid_logp < NetModel.llogp:
#                        NetModel.updateTheanoParams()
#                else:
#                        NetModel.updateNumpyParams()
#                        NetModel.saveNet()
                    # learning rate decay
#                if lr_divide>=NetModel.lr_divide:
#                        NetModel.lr *= NetModel.lr_decay
                    # early stopping
#                if NetModel.valid_logp*NetModel.min_impr<NetModel.llogp:
#                        if lr_divide<NetModel.lr_divide:
#                                NetModel.lr *= NetModel.lr_decay
#                                lr_divide += 1
#                        else:
#                                NetModel.saveNet()
#                                print 'Training completed.'
#                                break
                    # set last epoch objective value
#                NetModel.llogp = NetModel.valid_logp


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

#dmodel = Model()
#call_model = dmodel.initNet('encdec.cfg')
#training = Train(dmodel)



