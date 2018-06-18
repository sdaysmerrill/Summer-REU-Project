from data import *
from models import *
from test import *
from train import *
from plot_loss import *
from evaluate import *


output_words, attentions = evaluate("je suis trop froid .")
plt.matshow(attentions.numpy())

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def evaluate_and_show_attention(input_sentence):
    output_words, attentions = evaluate(input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)

evaluate_and_show_attention("elle a cinq ans de moins que moi .")

evaluate_and_show_attention("elle est trop petit .")

evaluate_and_show_attention("je ne crains pas de mourir .")

evaluate_and_show_attention("c est un jeune directeur plein de talent .")



