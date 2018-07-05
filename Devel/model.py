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
        
        self.embed_size = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, word_inputs, hidden, hidden_size):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embed_size(word_inputs).view(seq_len, 1, -1)
        
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA:hidden = hidden.cuda()
        return hidden
    
#    def setWordVec(self, word2vec):  #figure out where word2vec comes from
#        self.Wemb_np = self.Wemb.get_value()   #where is .get_value()
#        for w, v in word2vec.iteritems():
#            self.Wemb_np[w,:] = v
#        self.Wemb.set_value(self.Wemb_np)

#    def emb(self, a, s, v):
#        a_emb = torch.sum(self.Wah[a,:], axis =0)
#        s_emb = self.Wsh[s,:]
#        v_emb = self.Wvh[v,:]
#        sv_emb = s_emb + v_emb

#        return a_emb, sv_emb
    
 

#class LSTM(nn.Module):
#    def __init__(self, input_size, hidden_size, embed_size):#, output_size):
#        super(LSTM, self).__init__()

#        self.hidden_size = hidden_size
        # input embedding
 #       self.encoder = EncoderRNN(input_size, hidden_size)  ##nn.Embedding(input_size, embed_size)
        # lstm weights
#        self.weight_fm = nn.Linear(hidden_size, hidden_size)
#        self.weight_im = nn.Linear(hidden_size, hidden_size)
#        self.weight_cm = nn.Linear(hidden_size, hidden_size)
#        self.weight_om = nn.Linear(hidden_size, hidden_size)
#        self.weight_fx = nn.Linear(embed_size, hidden_size)
#        self.weight_ix = nn.Linear(embed_size, hidden_size)
#        self.weight_cx = nn.Linear(embed_size, hidden_size)
#        self.weight_ox = nn.Linear(embed_size, hidden_size)
        # multiplicative weights
#        self.weight_mh = nn.Linear(hidden_size, hidden_size)
#        self.weight_mx = nn.Linear(embed_size, hidden_size)
        # decoder
 #       self.decoder = DecoderRNN(hidden_size, output_size)###nn.Linear(hidden_size, output_size)


#    def forward(self, inp, h_0, c_0):
        # encode the input characters
#        inp = self.encoder(inp)
        # calculate the multiplicative matrix
#        m_t = self.weight_mx(inp) * self.weight_mh(h_0)
        # forget gate
#        f_g = F.sigmoid(self.weight_fx(inp) + self.weight_fm(m_t))
        # input gate
#        i_g = F.sigmoid(self.weight_ix(inp) + self.weight_im(m_t))
        # output gate
#        o_g = F.sigmoid(self.weight_ox(inp) + self.weight_om(m_t))
        # intermediate cell state
#        c_tilda = F.tanh(self.weight_cx(inp) + self.weight_cm(m_t))
        # current cell state
#        cx = f_g * c_0 + i_g * c_tilda
        # hidden state
#        hx = o_g * F.tanh(cx)

#        out = self.decoder(hx.view(1,-1))

#        return out, hx, cx

#    def init_hidden(self):
#        h_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
#        c_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
 #       return h_0, c_0

        

class WenAttn(nn.Module):
    def attend(self,hidden_size,attn):
        print("in attention")
        state_x = nn.Linear(hidden_size,attn)   
        score_x = nn.Tanh(nn.Linear(state_x))
        return score_x
    
    def __init__(self, hidden_size, max_length=MAX_LENGTH):
        super(WenAttn, self).__init__()
        
 #       self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)   #issue is that these are tensors

        self.attend(self.hidden_size,self.attn)
    

    
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
        #change to LSTM not GRU
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

 #       self.attn = WenAttn(hidden_size)
     
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        decoder_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        #change to LSTM
        decoder_output, hidden = self.gru(decoder_input, last_hidden)

        
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = WenAttn(encoder_outputs.squeeze(0))     #rnn_output.squeeze(0),
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
