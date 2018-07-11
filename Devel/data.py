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
from model import EncoderRNN
from loader.DataReader import *
from loader.GentScorer import *

from ConfigParser import SafeConfigParser

#USE_CUDA = False


class Model(object):

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
            else: # init a new model
                self.initNet(config,opts)
                self.updateNumpyParams()

    def initNet(self,config,opts=None):
            
            print '\n\ninit net from scrach ... '

            # config parser
            parser = SafeConfigParser()
            parser.read(config)

            # setting learning hyperparameters 
            self.debug = parser.getboolean('learn','debug')
            if self.debug:
                print 'loading settings from config file ...'
            self.seed       = parser.getint(  'learn','random_seed')
            self.lr_divide  = parser.getint(  'learn','lr_divide')
            self.lr         = parser.getfloat('learn','lr')
            self.lr_decay   = parser.getfloat('learn','lr_decay')
            self.beta       = parser.getfloat('learn','beta')
            self.min_impr   = parser.getfloat('learn','min_impr')
            self.llogp      = parser.getfloat('learn','llogp')
            # setting training mode
            self.mode       = parser.get('train_mode','mode')
            self.obj        = parser.get('train_mode','obj')
            self.gamma      = parser.getfloat('train_mode','gamma')
            self.batch      = parser.getint('train_mode','batch')    #changed getint to get
            # setting file paths
            if self.debug:
                print 'loading file path from config file ...'
            self.wvecfile   = parser.get('data','wvec')
            self.trainfile  = parser.get('data','train')
            self.validfile  = parser.get('data','valid') 
            self.testfile   = parser.get('data','test')
            self.vocabfile  = parser.get('data','vocab')
            self.domain     = parser.get('data','domain')
            self.percentage = float(parser.getfloat('data','percentage'))/100.0
            # Setting generation specific parameters
            self.topk       = parser.getint('gen','topk')
            self.overgen    = parser.getint('gen','overgen')
            self.beamwidth  = parser.getint('gen','beamwidth')
            self.detectpairs= parser.get('gen','detectpairs')
            self.verbose    = parser.getint('gen','verbose')
            self.decode     = parser.get('gen','decode')
            # setting rnn configuration
            self.gentype    = parser.get('generator','type')
            self.dh         = parser.getint('generator','hidden')  #changed getint to get
            # set random seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            np.set_printoptions(precision=4)
            # setting data reader, processors, and lexicon
            self.setupDelegates()
            # network size
            self.di = len(self.reader.vocab)
            # logp for validation set
            self.valid_logp = 0.0  
            # start setting networks 
            self.initModel()
            self.model.config_theano()

    def initModel(self):
        #################################################################
        #################### Model Initialisation #######################
        #################################################################
            if self.debug:
                print 'setting network structures using variables ...'
            ###########################################################
            ############## Setting Recurrent Generator ################
            ###########################################################
            if self.debug:
                print '\tsetting recurrent generator, type: %s ...' % \
                        self.gentype
            self.model =  EncoderRNN(self.di, self.dh) #NNGenerator(self.gentype, self.reader.vocab,
 #                   self.beamwidth, self.overgen,
#                    self.di, self.dh, self.batch, self.reader.dfs, 
#                    self.obj, self.mode, self.decode, 
#                    self.reader.tokenMap2Indexes()) 
            # setting word vectors
            if self.wvecfile!='None':
                self.model.setWordVec(self.reader.readVecFile(
                    self.wvecfile,self.reader.vocab))
            if self.debug:
                print '\t\tnumber of parameters : %8d' % \
                        self.model.numOfParams()
                print '\tthis may take up to several minutes ...'


    def setupDelegates(self):
            # initialise data reader
            self.reader = DataReader(self.seed, self.domain, self.obj,
                    self.vocabfile, self.trainfile, self.validfile, self.testfile,
                    self.percentage, self.verbose, lexCutoff=4)
            # setting generation scorer
            self.gentscorer = GentScorer(self.detectpairs)



