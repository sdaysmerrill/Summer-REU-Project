from data import *
import numpy as np

##MAX_LENGTH = 10

gentype =
vocab =
beamwidth = 10
overgen = 20
vocab_size =
hidden_size =
batch_size = 1
da_sizes =

class EncoderRNN():
 #   def __init__(self, input_size, hidden_size, n_layers=1):
#        super(EncoderRNN, self).__init__()
        
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.n_layers = n_layers
        
#        self.embedding = nn.Embedding(input_size, hidden_size)

        #change to LSTM
#        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
#    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
#        seq_len = len(word_inputs)
#        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        #change to LSTM
#        output, hidden = self.gru(embedded, hidden)
#        return output, hidden

#    def init_hidden(self):
#        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
#        if USE_CUDA:hidden = hidden.cuda()
#        return hidden

    def __init__(self, gentype, vocab, beamwidth, overgen, vocab_size,
                 hidden_size, batch_size, da_sizes):   
        BaseRLG.__init__(self, gentype, vocab, beamwidth, overgen, vocab_size,
                 hidden_size, batch_size, da_sizes)   
        self.da = self.dfs[1]-self.dfs[0]   #what is dfs??
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
        #nned to find a way to get around using numpy first
        self.Wah = 0.3 * np.random.uniform(-1.0,1.0,(self.da+1, self.dh)).astype(float)
        self.Wah = torch.from_numpy(self.Wah)
        self.Wsh = 0.3 * np.random.uniform(-1.0,1.0,(self.ds+1, self.dh)).astype(float)
        self.Wsh = torch.from_numpy(self.Wsh)
        self. Wvh = 0.3 * np.random.uniform(-1.0,1.0,(self.dv+1, self.dh)).astype(float)
        self.Wvh = torch.from_numpy(self.Wvh)
        
        #attention weights
        self.Wha = 0.3 * np.random.uniform(-1.0,1.0,(self.dh*3 self.dh)).astype(float)
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


    def setWordVec(self, word2vec):  #figure out where word2vec comes from
        self.Wemb_np = self.Wemb.get_value()   #where is .get_value()
        for w, v in word2vec.iteritems():
            self.Wemb_np[w,:] = v
        self.Wemb.set_value(self.Wemb_np)

    def emb(self, a, s, v):
        a_emb = torch.sum(self.Wah[a,:], axis =0)
        s_emb = self.Wsh[s,;]
        v_emb = self.Wvh[v,:]
        sv_emb = s_emb + v_emb
        return a_emb, sv_emb

    def unroll():
        pass

<<<<<<< HEAD


class WenAttn(nn.Module)
    def attend():
        state_x = torch.cat([h_tm1, sv_emb_x], axis = 1)
        score_x = torch.dot(torch.tanh(torch.dot(state_x, self.Wha)), self.Vha)
        return score_x



#class Attn(nn.Module):   #change to _attend from RNNLG
#    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
#        super(Attn, self).__init__()
=======
class WenAttn(nn.Module):
    
    
class Attn(nn.Module):   #change to _attend from RNNLG
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()
>>>>>>> e06e10af4df544f4b4f0dbf531efe0b8c0903df2
        
#        self.method = method
#        self.hidden_size = hidden_size
        
#        if self.method == 'general':
#            self.attn = nn.Linear(self.hidden_size, hidden_size)

#        elif self.method == 'concat':
#            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
#            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

#    def forward(self, hidden, encoder_outputs):
#        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
#        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
#        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
#        for i in range(seq_len):
#            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
<<<<<<< HEAD
#        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

=======
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)


    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy =torch.dot(hidden.view(-1), encoder_output.view(-1))
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy

 #   def _attend():
#        state_x = torch.cat([], axis = 1)
#        score_x = torch.dot(torch.tanh(torch.dot(state_x, self._____)), self.___)
#        return score_x
    
class WenAttnDecoderRNN(nn.Module):
>>>>>>> e06e10af4df544f4b4f0dbf531efe0b8c0903df2

#    def score(self, hidden, encoder_output):
#        if self.method == 'dot':
#            energy =torch.dot(hidden.view(-1), encoder_output.view(-1))
#        elif self.method == 'general':
#            energy = self.attn(encoder_output)
#            energy = torch.dot(hidden.view(-1), energy.view(-1))
#        elif self.method == 'concat':
#            energy = self.attn(torch.cat((hidden, encoder_output), 1))
#            energy = torch.dot(self.v.view(-1), energy.view(-1))
#        return energy


    

class WenAttnDecoderRNN(nn.Module)
    def recur():
        #attention

        #compute ig,fg,og together and slice it
        gates_t = torch.dot(torch.cat([prev_hidden_state, da_emb_t], axis=1), self.Wgate)
        #compute all the gates
        ig = nn.Sigmoid(gates_t[:,:self.dh])
        fg = nn.Sigmoid(gates_t[:, self.dh:self.dh*2])
        og = nn.Sigmoid(gates_t[:,self.dh*2:self.dh*3])
        cx_t = nn.Tanh(gates_t[:,self.dh*3:])
        #update interal LSTM state
        c_t = ig*cx_t + fg*c_tm1
        #new hidden state
        h_t = og* nn.Tanh(c_t)
        #output and probability of target word
        o_t = nn.Softmax(torch.dot(h_t, self.Who)
        p_t = o_t[torch.arrange(self.db),y_t]

        return h_t, c_t, p_t

#class AttnDecoderRNN(nn.Module):
#    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
#        super(AttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
#        self.attn_model = attn_model
#        self.hidden_size = hidden_size
#        self.output_size = output_size
#        self.n_layers = n_layers
#        self.dropout_p = dropout_p
        
        # Define layers
#        self.embedding = nn.Embedding(output_size, hidden_size)
        #change to LSTM
#        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
#        self.out = nn.Linear(hidden_size * 2, output_size)
        
        # Choose attention model
#        if attn_model != 'none':
#            self.attn = Attn(attn_model, hidden_size)
    
#    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
#        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
#        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        #change to LSTM
#        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
#        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
#        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
#        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
#        context = context.squeeze(1)       # B x S=1 x N -> B x N
#        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
#        return output, context, hidden, attn_weights


