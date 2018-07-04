#####################################################################
##        Sarah Days-Merrill  Louisiana State University           ##
##             CCT REU     Mentor: Dr. Joohyun Kim                 ##
#####################################################################


import argparse
from data import *
from data import variables_from_pair
from model import *
from train import *


#USE_CUDA = False   this is set in data.py

teacher_forcing_ratio = 0.5
clip = 5.0


training_pair = variables_from_pair(random.choice(pairs))
input_variable = training_pair[0]
target_variable = training_pair[1]


hidden_size = 500
n_layers = 2        #so this is a 2 layer model? - so 2 layers of hidden states
dropout_p = 0.05

# Initialize model
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = DecoderRNN(hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)


# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
learning_rate = 0.0001
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()



def testNet():
    encoder_test = EncoderRNN(10, 10, 2)
    decoder_test = AttnDecoderRNN(10, 10, 2)   #'general',
    print(encoder_test)
    print(decoder_test)

    encoder_hidden = encoder_test.init_hidden()
    word_input = Variable(torch.LongTensor([1, 2, 3]))
    if USE_CUDA:
        encoder_test.cuda()
        word_input = word_input.cuda()
    encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

    word_inputs = Variable(torch.LongTensor([1, 2, 3]))
    decoder_attns = torch.zeros(1, 3, 3)
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size))

    if USE_CUDA:
        decoder_test.cuda()
        word_inputs = word_inputs.cuda()
        decoder_context = decoder_context.cuda()

    for i in range(3):
        decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_inputs[i], decoder_context, decoder_hidden, encoder_outputs)
        print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
        decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data







#this is the end of main file
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    args = parser.parse_args()

    
    if args.mode == 'train':
        print("Begin training")
        trainNet(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH)
        
    elif args.mode == 'test':
        print("Start testing")
        testNet()  

    else:
        print("Please specify a mode")
        






   






