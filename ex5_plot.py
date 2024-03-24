from embeddings import Embeddings
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np

def tsne_plot(corpus, model):
    labels = []
    tokens = []
    print("in func")
    for word in model.get_vocab().key_to_index:

        if word in stopwords.words('french'):
            continue
        tokens.append(model.get_vector(word))
        labels.append(word)
        #print(model.get_vocab()[word])

    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(np.array(tokens))

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title(corpus)
    plt.show()

corpus_path = './data_embeddings/GAUCHE.txt'
# model_path = './models/GAUCHE.W2Vmodel'
model_path = './models_correct/GAUCHE_restrictive.W2Vmodel'
emb = Embeddings(corpus_path, model_path)
emb.learn_restrictive(min_freq=1200)
emb.load()
model = emb.get_model()
tsne_plot('GAUCHE', emb)

corpus_path = './data_embeddings/GAUCHE.txt'
# model_path = './models/GAUCHE.W2Vmodel'
model_path = './models_correct/GAUCHE_restrictive.W2Vmodel'
emb = Embeddings(corpus_path, model_path)
emb.learn_restrictive(min_freq=1200)
emb.load()
model2 = emb.get_model()
tsne_plot('GAUCHE', emb)

print(model == model2)

corpus_path = './data_embeddings/DROITE.txt'
# model_path = './models/DROITE.W2Vmodel'
model_path = './models_correct/DROITE_restrictive.W2Vmodel'
emb = Embeddings(corpus_path, model_path)
emb.learn_restrictive(min_freq=1200)
emb.load()
#model = emb.get_model()
tsne_plot('DROITE', emb)
