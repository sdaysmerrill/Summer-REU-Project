#####################################################################
##                    Sarah Days-Merrill                           ##
##             CCT REU Louisiana State University                  ##
#####################################################################


from main import *


MAX_LENGTH = 10

def trainNet(input_variable, target_variable, encdec, encdec_optimizer, criterion, max_length=MAX_LENGTH):

    # Zero gradients of both optimizers
    encdec_optimizer.zero_grad()

    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden, hidden_size)  
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, encoder.hidden_size))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
 #   use_teacher_forcing = random.random() < teacher_forcing_ratio
#    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
 #       for di in range(target_length):
 #           decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
 #           loss += criterion(decoder_output, target_variable[di])
 #           decoder_input = target_variable[di] # Next target is next input

   # else:
        # Without teacher forcing: use network's own prediction as the next input
    for di in range(target_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_variable[di])
            
            # Get most likely word index (highest value) from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
            
        decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
        if USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
        if ni == EOS_token: break

    # Backpropagation
    clip = 5.0
    loss.backward()
    torch.nn.utils.clip_grad_norm(encdec.parameters(), clip)
 #   torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
  # torch.nn.utils.clip_grad_norm_.torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
  # torch.nn.utils.clip_grad_norm_.torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    
    encdec_optimizer.step()

    
    return loss.data / target_length




    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


    # Configuring training
    n_epochs = 500    #this is supposed to be 50000
    plot_every = 250  #200
    print_every = 100 #1000

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every


    #input_varaible = variables_from_pair.input_variable
    #target_variable = variables_from_pair.target_variable

    # Begin!
    for epoch in range(1, n_epochs + 1):
        
        # Get training data for this cycle
        training_pair = variables_from_pair(random.choice(pairs))
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # Run the train function
        loss = trainNet(input_variable, target_variable, encdec, encdec_optimizer, criterion)

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch *1.0 / n_epochs * 1.0), epoch, epoch*1.0 / n_epochs*1.0 * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
   

