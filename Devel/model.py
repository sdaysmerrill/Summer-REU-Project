#####################################################################
##        Sarah Days-Merrill  Louisiana State University           ##
##             CCT REU     Mentor: Dr. Joohyun Kim                 ##
#####################################################################


from data import *


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN,self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, word_inputs, hidden, hidden_size):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
 #       return embedded

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA:hidden = hidden.cuda()
        return hidden
    
    def setWordVec(self, word2vec):  #figure out where word2vec comes from
        self.Wemb_np = self.Wemb.get_value()   #where is .get_value()
        for w, v in word2vec.iteritems():
            self.Wemb_np[w,:] = v
        self.Wemb.set_value(self.Wemb_np)

    def emb(self, a, s, v):
        a_emb = torch.sum(self.Wah[a,:], axis =0)
        s_emb = self.Wsh[s,:]
        v_emb = self.Wvh[v,:]
        sv_emb = s_emb + v_emb

        return a_emb, sv_emb
    
 #   def lstm():
        #attention

        #compute ig,fg,og together and slice it
#        gates_t = torch.dot(torch.cat([prev_hidden_state, da_emb_t], axis=1), self.Wgate)
        #compute all the gates
#        ig = nn.Sigmoid(gates_t[:,:self.dh])
#        fg = nn.Sigmoid(gates_t[:, self.dh:self.dh*2])
#        og = nn.Sigmoid(gates_t[:,self.dh*2:self.dh*3])
#        cx_t = nn.Tanh(gates_t[:,self.dh*3:])
        #update interal LSTM state
#        c_t = ig*cx_t + fg*c_tm1
        #new hidden state
#        h_t = og* nn.Tanh(c_t)
        #output and probability of target word
#        o_t = nn.Softmax(torch.dot(h_t, self.Who)
#        p_t = o_t*[torch.arrange(self.db),y_t]   #something wrong with this line
                         #invalid syntax here too

#        return h_t, c_t, p_t


class WenAttn(nn.Module):
    def __init__(self, hidden_size, max_length=MAX_LENGTH):
        super(WenAttn, self).__init__()
        
 #       self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)
    
 #   def attend(hidden_size,attn):
#        state_x = torch.cat([hidden_size, attn], axis = 1)
#        score_x = q * nn.Tanh(nn.Linear(state_x))
#        return score_x



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout_p):   #n_layers = 1, dropout_p=0.1
        super(DecoderRNN, self).__init__()
        
        # Keep parameters for reference
 #       self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        #change to LSTM
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        self.attn = WenAttn(hidden_size)
        # Choose attention model
 #       if attn_model != 'none':
         #attn_model, hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        decoder_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        #change to LSTM
        decoder_output, hidden = self.gru(decoder_input, last_hidden)

        
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(encoder_outputs.squeeze(0))     #rnn_output.squeeze(0),
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        decoder_output = decoder_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((decoder_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


#class Model(nn.Module):
#    def __int__(self):
#        if True:
#            self.op = EncoderRNN()
#        else:
#            self.op = DecoderRNN()

#    def forward(self, input):
#        return self.op(input)
