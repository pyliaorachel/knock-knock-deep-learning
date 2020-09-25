import os.path

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def display_pca_scatterplot(model, words):
    # Take word vectors
    word_vectors = np.array([model[w] for w in words])

    # PCA, take the first 2 principal components
    twodim = PCA().fit_transform(word_vectors)[:,:2]

    # Draw
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
    plt.show()

def main():
    # Download pre-trained GloVe embeddings, turn into Word2Vec format
    glove_file = './data/glove.6B.100d.txt'
    word2vec_glove_file = './data/glove.6B.100d.word2vec.txt'
    if not os.path.isfile(word2vec_glove_file):
        glove2word2vec(glove_file, word2vec_glove_file)

    # Load model
    model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

    # Take word, return most similar words
    word = input('Similarity - Input word: ')
    while word != '':
        print(model.most_similar(word))
        word = input('Similarity - Input word: ')

    # Take a, b, c, return d for a : b = c : d
    a = input('Analogy a : b = c : d - Input a: ')
    b = input('Analogy a : b = c : d - Input b: ')
    c = input('Analogy a : b = c : d - Input c: ')
    while a != '' and b != '' and c != '':
        print(model.most_similar(positive=[c, b], negative=[a]))
        a = input('Analogy a : b = c : d - Input a: ')
        b = input('Analogy a : b = c : d - Input b: ')
        c = input('Analogy a : b = c : d - Input c: ')

    # Display scatter plot for words
    choice = input('2D word vector visualization - Input \'default\' for a default list of words, or a comma-separated list of words: ')
    while choice != '':
        if choice == 'default':
            words = ['coffee', 'tea', 'beer', 'wine', 'brandy', 'rum', 'champagne', 'water',
                     'spaghetti', 'borscht', 'hamburger', 'pizza', 'falafel', 'sushi', 'meatballs',
                     'dog', 'horse', 'cat', 'monkey', 'parrot', 'koala', 'lizard',
                     'frog', 'toad', 'monkey', 'ape', 'kangaroo', 'wombat', 'wolf',
                     'france', 'germany', 'hungary', 'luxembourg', 'australia', 'fiji', 'china',
                     'homework', 'assignment', 'problem', 'exam', 'test', 'class',
                     'school', 'college', 'university', 'institute']
            display_pca_scatterplot(model, words=words)
        else:
            display_pca_scatterplot(model, words=[word.strip() for word in choice.split(',')])
        choice = input('2D word vector visualization - Input \'default\' for a default list of words, or a comma-separated list of words: ')

if __name__ == '__main__':
    main()
