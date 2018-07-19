#####################################################################
##                    Sarah Days-Merrill                           ##
##             CCT REU Louisiana State University                  ##
#####################################################################


import data


import torch
import torch.nn as nn
from torch.autograd import Variable


MAX_LENGTH = 10
USE_CUDA = False
dropout_p = 0.05




def setWordVec(NetModel, word2vec):   
    Wemb_np = NetModel.Wemb.get_value()
    for w,v in word2vec.iteritems():
        Wemb_np[w,:] = v
    NetModel.Wemb.set_value(Wemb_np)

def _emb(NetModel, a, s, v):
    a_emb = torch.sum(NetModel.Wah[a,:],dim=0)  #sum over the rows
    s_emb = NetModel.Wsh[s,:]
    v_emb = NetModel.Wvh[v,:]
    sv_emb= s_emb + v_emb
    return a_emb, sv_emb

def embedding(NetModel, a,s,v,words):
        
    # embed DA
    a_emb,sv_emb= _emb(NetModel, a,s,v)
    words_emb = setWordVec(NetModel, words)
 #       sv_emb = sv_emb.dimshuffle(1,0,2)
        # recurrence
 #       [h,c,p],_ = theano.scan(fn=self._recur,
#                sequences=[words[:-1,:],words[1:,:]],
#                outputs_info=[self.h0,self.c0,None],
#                non_sequences=[a_emb,sv_emb])
        # compute desired sent_logp by slicing
#        cutoff_logp = collectSentLogp(p,cutoff_f[4],cutoff_b)
#        cost = -T.sum(cutoff_logp)
    print("(debug) this is what a_emb holds ",a_emb)
    print("(debug) this is what s_emb holds ",sv_emb)
    return a_emb, sv_emb ,words_emb

 #   def forward(self, tensor_a, tensor_s, tensor_v):
        # Note: we run this all at once (over the whole input sequence)
 #       seq_len = len(word_inputs)
#        embedded = self.setWordVec(word_inputs).view(seq_len, 1, -1)
 #       output, hidden = self.gru(embedded, hidden)
#         a_emb,s_emb, v_emb = self._emb(tensor_a, tensor_s, tensor_v)
#         print("(debug) this is what a_emb holds ",a_emb)
#         print("(debug) this is what s_emb holds ",s_emb)
#         print("(debug) this is what v_emb holds ",v_emb)
#         return a_emb, s_emb, v_emb      

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

    
    
 
#still need to incorporate this into the model in the decoder
class WenLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, Wgate, Who):  
        super(WenLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wgate = Wgate
        self.Who = Who

      
        
        # input embedding
 #       self.encoder = EncoderRNN(self.input_size, self.hidden_size) 
        # lstm weights
   #     self.weight_fm = nn.Linear(hidden_size, hidden_size)
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
 #       self.decoder = DecoderRNN(self.input_size, self.hidden_size, n_layers = 2, dropout_p = dropout_p)


    def forward(self, wv_t, h_tm1, da_emb_t, h_0, c_0):
        
        gates_t = torch.dot(torch.cat((wv_t, h_tm1, da_emb_t), 1), self.Wgate)  #concatenate along axis = 1
        #compute gates
        ig = nn.Sigmoid(gates_t[:,:self.hidden_size])
        fg = nn.Sigmoid(gates_t[:, self.hidden_size:self.hidden_size * 2])
        og = nn.Sigmoid(gates_t[:,self.hidden_size * 2: self.hidden_size * 3])
        cx_t = nn.Tanh(gates_t[:,self.hidden_size * 3:])
        #update lstm internal state
        c_t = ig * cx_t + fg * c_tm1
        #obtain new hidden layer
        h_t = og * nn.Tanh(c_t)
        #compute output distribution target word probability
        o_t = nn.Softmax(torch.dot(h_t, self.Who))
        p_t = o_t[torch.arange(NetModel.db), y_t]  #where does db come from??

        return h_t, c_t, p_t
        
        # encode the input characters
   #     inp = self.encoder(inp)
        # calculate the multiplicative matrix
 #       m_t = self.weight_mx(inp) * self.weight_mh(h_0)
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

    def init_hidden(self):   #do I need another init_hidden?? - well I need to initialize the context vector
        h_0 = Variable(torch.zeros(1, self.hidden_size))
        c_0 = Variable(torch.zeros(1, self.hidden_size))
        if USE_CUDA:
            h_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
            c_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        return h_0, c_0

        

class WenAttn(nn.Module):
    def __init__(self, hidden_size, max_length=MAX_LENGTH):
        super(WenAttn, self).__init__()
        
        self.hidden_size = hidden_size
 #       self.attn = nn.Linear(self.hidden_size, hidden_size)   

 #      
    def forward(self,hidden,sv_emb):
        seq_len = len(sv_emb)

        # Create variable to store attention scores
        attn_scores = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA: attn_scores = attn_scores.cuda()

        # Calculate scores for each encoder output
        for i in range(seq_len):
            attn_scores[i] = self.attend(hidden, sv_emb[i])

        # Normalize scores to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_scores).unsqueeze(0).unsqueeze(0)


    def attend(self, a_emb, sv_emb):
        score_x = nn.Linear(a_emb, sv_emb)    #need another agrument - would it be a_emb?
        score_x = hidden.dot(score_x)  #what is hidden.dot?
        
 #       state_x = nn.Linear(hidden_size,attn)   
#        score_x = nn.Tanh(nn.Linear(state_x))
        return score_x

   
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_p):    
        super(DecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        #lstm weighted matrices
        self.Wgate = 0.3 * torch.randn(self.hidden_size * 3, self.hidden_size * 4, dtype = torch.float64)
        self.Who = 0.3 * torch.randn(self.hidden_size, self.input_size, dtype = torch.float64)


        self.Wgate = torch.clamp(self.Wgate, min = -1.0, max = 1.0)
        self.Who = torch.clamp(self.Who, min = -1.0, max = 1.0)

        self.Wgate_parameter = nn.Parameter(self.Wgate)
        self.Who_parameter = nn.Parameter(self.Who)
        # Define layers
 #       self.embedding = nn.Embedding(output_size, hidden_size)
        #change to LSTM not GRU
        self.lstm = WenLSTM(self.input_size, self.hidden_size, self.Wgate, self.Who) #, dropout=dropout_p)
 #       self.out = nn.Linear(hidden_size * 2, output_size)
 #       self.softmax = nn.LogSoftmax(dim=1)
 #       self.cross_entrophy = nn.CrossEntropyLoss()

        self.attn = WenAttn(hidden_size)
     
    
    def forward(self, a_emb, sv_emb, words_emb):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
   #     word_embedded = nn.Embedding(input).view(1, 1, -1) # S=1 x B x N

       #input words embedding
 #      wv_t =

        attn_weights = self.attn(a_emb, sv_emb)

        sv_emb_t = torch.dot(attn_weights, sv_emb)
        da_emb_t = nn.Tanh(a_emb + sv_emb_t)
        # Combine embedded input word and last context, run through RNN
 #      decoder_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        #change to LSTM
        decoder_output, hidden = self.lstm(wv_t, h_tm1, da_emb_t)
        
        
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
 #       attn_weights = WenAttn(encoder_outputs.squeeze(0))     #rnn_output.squeeze(0),
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        decoder_output = decoder_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = self.cross_entrophy(self.out(torch.cat((decoder_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


#class EncDecRNN(nn.Module):
#    def __init__(self, input_size, hidden_size, n_layers, dropout_p = dropout_p):
#        super(EncDecRNN, self).__init__()
#        self.input_size = input_size
#        self.hidden_size = hidden_size
        
#        self.n_layers = n_layers
#        self.dropout_p = dropout_p
        
#        self.encoder = EncoderRNN(self.input_size, self.hidden_size)  
#        self.decoder = DecoderRNN(self.input_size, self.hidden_size, self.n_layers, self.dropout_p)
  

#    def forward(self):
        #call EncoderRNN
#        enc_hidden = self.encoder(self.input_size, self.hidden_size, self.n_layers)
        #call DecoderRNN
#        dec_output, dec_context, dec_hidden, dec_attn_weights = self.decoder(enc_hidden, hidden_size, n_layers, dropout_p=dropout_p)
#        return dec_output, dec_context, dec_hidden, dec_attn_weights

   
