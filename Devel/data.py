#####################################################################
##                    Sarah Days-Merrill                           ##
##             CCT REU Louisiana State University                  ##
#####################################################################


import unicodedata
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

#from NNGenerator import *
from model import DecoderRNN
from train import trainNet

from loader.DataReader import *
from loader.GentScorer import *

from ConfigParser import SafeConfigParser

#USE_CUDA = False
##hidden_size = 500
n_layers = 2
dropout_p = 0.05
##MAX_LENGTH = 10

class NetModel(object):

    #######################################################################
    # all variables that needs to be save and load from model file, indexed 
    # by their names
    #######################################################################
    params_vars = ['self.params_np']
    learn_vars  = ['self.lr','self.lr_decay','self.beta','self.seed',
            'self.min_impr','self.llogp', 'self.debug','self.valid_logp',
            'self.lr_divide']
    mode_vars   = ['self.mode','self.obj','self.gamma','self.batch']
    data_vars   = ['self.domain', 'self.wvecfile', 'self.modelfile',
            'self.vocabfile', 'self.trainfile','self.validfile','self.testfile',
            'self.percentage']
    gen_vars    = ['self.topk','self.beamwidth','self.overgen',
            'self.detectpairs','self.verbose']
    model_vars  = ['self.gentype','self.di','self.dh']

    def __init__(self,config=None,opts=None):
        #config is defined by the command line arguments
        #it should be encdec.cfg
            # not enough info to execute
            if config==None and opts==None:
                print "Please specify command option or config file ..."
                return
            # config parser
            parser = SafeConfigParser()
            parser.read(config)
            # loading pretrained model if any
            self.modelfile = parser.get('data','model')
            if opts:    self.mode = opts.mode
            # check model file exists or not 
            if os.path.isfile(self.modelfile):
                if not opts:    self.loadNet(parser,None)
                else:           self.loadNet(parser,opts.mode)
            else: # init a new model - this gets called since above is False
                self.initNet(config,opts)
 #               self.updateNumpyParams()

    def initNet(self,config,opts=None):
            
            print '\n\ninit net from scratch ... '

            # config parser
            parser = SafeConfigParser()
            parser.read(config)

            # setting learning hyperparameters
            #this is set as True so self.debug holds 'True'
            self.debug = parser.getboolean('learn','debug')
            if self.debug:
                print 'loading settings from config file ...'
            self.seed       = parser.getint(  'learn','random_seed') #set as 5
            self.lr_divide  = parser.getint(  'learn','lr_divide')   #3
            self.lr         = parser.getfloat('learn','lr')          #0.1
            self.lr_decay   = parser.getfloat('learn','lr_decay')    #0.5
            self.beta       = parser.getfloat('learn','beta')        #0.0000001
            self.min_impr   = parser.getfloat('learn','min_impr')    #1.003
            self.llogp      = parser.getfloat('learn','llogp')       #-100000000
            # setting training mode
            self.mode       = parser.get('train_mode','mode')        #all
            self.obj        = parser.get('train_mode','obj')         #ml
            self.gamma      = parser.getfloat('train_mode','gamma')  #5.0
            self.batch      = parser.getint('train_mode','batch')    #1  
            # setting file paths
            if self.debug:
                print 'loading file path from config file ...'
            self.wvecfile   = parser.get('data','wvec')             #vec/vectors-80.txt
            self.trainfile  = parser.get('data','train')            #data/original/restuarant/train.json
            self.validfile  = parser.get('data','valid')            #data/original/restuarant/valid.json
            self.testfile   = parser.get('data','test')             #data/original/restuarant/test.json
            self.vocabfile  = parser.get('data','vocab')            #resource/vocab
            self.domain     = parser.get('data','domain')           #restuarant
            self.percentage = float(parser.getfloat('data','percentage'))/100.0   #100/100.0 = 1.0
            # Setting generation specific parameters
            self.topk       = parser.getint('gen','topk')           #5
            self.overgen    = parser.getint('gen','overgen')        #20
            self.beamwidth  = parser.getint('gen','beamwidth')      #10
            self.detectpairs= parser.get('gen','detectpairs')       #resource/detect.pair
            self.verbose    = parser.getint('gen','verbose')        #1
            self.decode     = parser.get('gen','decode')            #beam
            # setting rnn configuration
            self.gentype    = parser.get('generator','type')        #encdec
            self.dh         = parser.getint('generator','hidden')   #80
            # set random seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            np.set_printoptions(precision=4)
            # setting data reader, processors, and lexicon
            self.setupDelegates()    #setupDelegates is defined below
            # network size
            self.di = len(self.reader.vocab)   #length of input - input_size
            # logp for validation set

            self.dfs = self.reader.dfs                  #[0, 15, 149, 191, 200] - dimension of feature size

            self.da = self.dfs[1] - self.dfs[0]   #15
            self.ds = self.dfs[3] - self.dfs[2]   #42
            self.dv = self.dfs[4] - self.dfs[3]   #9

            #define random weighted matrices
            self.Wah = 0.3 * torch.randn(self.da+1, self.dh, dtype = torch.float64)
            self.Wsh = 0.3 * torch.randn(self.ds+1, self.dh, dtype = torch.float64)
            self.Wvh = 0.3 * torch.randn(self.dv+1, self.dh, dtype = torch.float64)
            self.Wemb = 0.3 * torch.randn(self.di, self.dh, dtype = torch.float64)

            self.Wah[self.da,:] = 0.0
            self.Wsh[self.ds,:] = 0.0
            self.Wvh[self.dv,:] = 0.0

            #set boundaries for the weighted matrices
            self.Wah = torch.clamp(self.Wah, min = -1.0, max = 1.0)
            self.Wsh = torch.clamp(self.Wsh, min = -1.0, max = 1.0)
            self.Wvh = torch.clamp(self.Wvh, min = -1.0, max = 1.0)
            self.Wemb = torch.clamp(self.Wemb, min = -1.0, max = 1.0) 

        #specify emodel's parameters for optimization
 #       self.Wah_parameter = nn.Parameter(self.Wah)
 #       self.Wsh_parameter = nn.Parameter(self.Wsh)
 #       self.Wvh_parameter = nn.Parameter(self.Wvh)
 #       self.Wemb_parameter = nn.Parameter(self.Wemb)
            
            self.valid_logp = 0.0  
            # start setting networks 
            self.initModel()
 #           self.model.config_theano()

    def initModel(self):
        #################################################################
        #################### Model Initialisation #######################
        #################################################################
            if self.debug:   #this is set as True
                print 'setting network structures using variables ...'
            ###########################################################
            ############## Setting Recurrent Generator ################
            ###########################################################
            if self.debug:
                print '\tsetting recurrent generator, type: %s ...' % \
                        self.gentype
                #call the EncDecRNN model in model.py with parameters input_size, hidden_size, n_layers, and dropout_p
 #           self.emodel =  EncoderRNN(self.di,self.dh,self.reader.dfs, n_layers) #NNGenerator(self.gentype, self.reader.vocab,
 #                   self.beamwidth, self.overgen,
#                    self.di, self.dh, self.batch, self.reader.dfs, 
#                    self.obj, self.mode, self.decode, 
#                    self.reader.tokenMap2Indexes())
            self.dmodel = DecoderRNN(self.di, self.dh, n_layers, dropout_p)
            # setting word vectors
 #           if self.wvecfile!='None':
#                self.emodel.setWordVec(self.reader.readVecFile(
#                    self.wvecfile,self.reader.vocab))
#            if self.debug:
#                print '\t\tnumber of parameters : %8d' % \
#                        self.model.numOfParams()
#                print '\tthis may take up to several minutes ...'


    def setupDelegates(self):
            # initialize data reader
            self.reader = DataReader(self.seed, self.domain, self.obj,
                    self.vocabfile, self.trainfile, self.validfile, self.testfile,
                    self.percentage, self.verbose, lexCutoff=4)
            # setting generation scorer
            self.gentscorer = GentScorer(self.detectpairs)



