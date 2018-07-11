#####################################################################
##                    Sarah Days-Merrill                           ##
##             CCT REU Louisiana State University                  ##
#####################################################################


from data import *
import numpy as np

MAX_LENGTH = 10
USE_CUDA = False
dropout_p = 0.05

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
 #       self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
 #       output, hidden = self.gru(output, hidden)
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


    def setWordVec(self, word2vec):  
        self.Wemb_np = self.Wemb.get_value()   
        for w, v in word2vec.iteritems():
            self.Wemb_np[w,:] = v
        self.Wemb.set_value(self.Wemb_np)
   

    def emb(self, a, s, v):
        a_emb = torch.sum(self.Wah[a,:], axis =0)
        s_emb = self.Wsh[s,:]
        v_emb = self.Wvh[v,:]
        sv_emb = s_emb + v_emb

        return a_emb, sv_emb
    
 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size):  #, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        # input embedding
        self.encoder = EncoderRNN(input_size, hidden_size)  ##nn.Embedding(input_size, embed_size)
        # lstm weights
        self.weight_fm = nn.Linear(hidden_size, hidden_size)
        self.weight_im = nn.Linear(hidden_size, hidden_size)
        self.weight_cm = nn.Linear(hidden_size, hidden_size)
        self.weight_om = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(embed_size, hidden_size)
        self.weight_ix = nn.Linear(embed_size, hidden_size)
        self.weight_cx = nn.Linear(embed_size, hidden_size)
        self.weight_ox = nn.Linear(embed_size, hidden_size)
        # multiplicative weights
        self.weight_mh = nn.Linear(hidden_size, hidden_size)
        self.weight_mx = nn.Linear(embed_size, hidden_size)
        # decoder
        self.decoder = DecoderRNN(hidden_size, output_size)###nn.Linear(hidden_size, output_size)


    def forward(self, inp, h_0, c_0):
        # encode the input characters
        inp = self.encoder(inp)
        # calculate the multiplicative matrix
        m_t = self.weight_mx(inp) * self.weight_mh(h_0)
        # forget gate
        f_g = F.sigmoid(self.weight_fx(inp) + self.weight_fm(m_t))
        # input gate
        i_g = F.sigmoid(self.weight_ix(inp) + self.weight_im(m_t))
        # output gate
        o_g = F.sigmoid(self.weight_ox(inp) + self.weight_om(m_t))
        # intermediate cell state
        c_tilda = F.tanh(self.weight_cx(inp) + self.weight_cm(m_t))
        # current cell state
        cx = f_g * c_0 + i_g * c_tilda
        # hidden state
        hx = o_g * F.tanh(cx)

        out = self.decoder(hx.view(1,-1))

        return out, hx, cx

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(1, self.hidden_size)).cuda()
        return h_0, c_0

        

class WenAttn(nn.Module):
    def __init__(self, hidden_size, max_length=MAX_LENGTH):
        super(WenAttn, self).__init__()
        
 #       self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)   #issue is that these are tensors

 #       self.attend(self.hidden_size,self.attn)

    def forward(self,hidden,encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_scores = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA: attn_scores = attn_scores.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_scores[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_scores).unsqueeze(0).unsqueeze(0)

    def attend(self,hidden_size, encoder_output):
        score_x = self.attn(encoder_output)
        score_x = hidden.dot(score_x)
        
 #       state_x = nn.Linear(hidden_size,attn)   
#        score_x = nn.Tanh(nn.Linear(state_x))
        return score_x

   
    
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
        self.softmax = nn.LogSoftmax(dim=1)

        self.attn = WenAttn(hidden_size)
     
    
    def forward(self, input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(input).view(1, 1, -1) # S=1 x B x N
        
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
        output = self.softmax(self.out(torch.cat((decoder_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


class EncDecRNN(nn.Module):
    def __int__(self, input_size, hidden_size, n_layers, dropout_p = dropout_p):
 
 #   def __init__(self):   #, gentype, vocab, beamwidth, overgen, vocab_size,
 #                hidden_size, batch_size, da_sizes): 

        super(EncDecRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        self.encoder = EncoderRNN(self.input_size, self.hidden_size, self.n_layers)
        self.decoder = DecoderRNN(self.hidden_size, self.output_size, self.n_layers, self.dropout_p)
  

        self.da = self.dfs[1]-self.dfs[0]   
        self.ds = self.dfs[3]-self.dfs[2]
        self.dv = self.dfs[4]-self.dfs[3]

        self.init_params()

    def init_params(self):
        #word embedding weight matrix   - where does di and dh come from??
        #self.di and self.dh give the dimensions of the tensor
        self.Wemb = 0.3 * np.random.uniform(-1.0,1.0, (self.di, self.dh)).astype(float)
        self.Wemb = torch.from_numpy(self.Wemb)
        #torch.rand.uniform(-1.0,1.0,(self.di, self.dh)).astype(floatX) *0.3

        #DA embedding
        #need to find a way to get around using numpy first
        self.Wah = 0.3 * np.random.uniform(-1.0,1.0,(self.da+1, self.dh)).astype(float)
        self.Wah = torch.from_numpy(self.Wah)
        self.Wsh = 0.3 * np.random.uniform(-1.0,1.0,(self.ds+1, self.dh)).astype(float)
        self.Wsh = torch.from_numpy(self.Wsh)
        self. Wvh = 0.3 * np.random.uniform(-1.0,1.0,(self.dv+1, self.dh)).astype(float)
        self.Wvh = torch.from_numpy(self.Wvh)
        
        #attention weights
        self.Wha = 0.3 * np.random.uniform(-1.0,1.0,(self.dh*3, self.dh)).astype(float)
        self.Wha = torch.from_numpy(self.Wha)
        self.Vha = 0.3 * np.random.uniform(-1.0,1.0,(self.dh)).astype(float)
        self.Vha = torch.from_numpy(self.Vha)

        #LSTM gate matrix
        self.Wgate = 0.3 * np.random.uniform(-1.0,1.0,(self.dh*3, self.dh)*4).astype(float)
        self.Wgate = torch.from_numpy(self.Wgate)

        #hidden to output matrix
        self.Who = 0.3 * np.random.uniform(-1.0,1.0,(self.dh, self.di)).astype(float)
        self.Who = torch.from_numpy(self.Who)

        #initialize the hidden state and cell
        self.h0 = torch.zeros(self.db,self.dh, dtype = float)
        self.c0 = torch.zeros(self.db,self.dh, dtype = float)


    def forward(self, input_size, hidden_size, n_layers, dropout_p = dropout_p):
        enc_output = self.encoder(input_size, hidden_size, n_layers)
        dec_output, dec_context, dec_hidden, dec_attn_weights = self.decoder(hidden_size, output_size, n_layers, dropout_p=dropout_p)
        return dec_output, dec_context, dec_hidden, dec_attn_weights

   
