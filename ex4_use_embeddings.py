from embeddings import Embeddings
import numpy as np
from ex3_test_embeddings import ex3_test_embeddings

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos = dot / (norm1 * norm2)
    return cos
    

droite_path = './data_embeddings/DROITE.txt'
gauche_path = './data_embeddings/GAUCHE.txt'

model_dr_path = './models/DROITE.W2Vmodel'
model_ga_path = './models/GAUCHE.W2Vmodel'

embeddings_dr = Embeddings(droite_path, model_dr_path)
embeddings_dr.load()

embeddings_ga = Embeddings(gauche_path, model_ga_path)
embeddings_ga.load()

embeddings = {"DROITE": embeddings_dr, "GAUCHE": embeddings_ga}

print("\n")
print('*'*80)
print("\nExercice 3 : Tests\n")
print('*'*80)
print("\n")
ex3_test_embeddings(embeddings)

print("\n\n\n")
print('*'*80)
print("\nExercice 4 : Utilisation des embeddings\n")
print('*'*80)
print("\n")

### Partie 1 : Étude des vecteurs les plus proches et éloignés entre les deux corpus

print("\n\n\n")
print('*'*80)
print("\nPartie 1 : Étude des vecteurs les plus proches et éloignés entre les deux corpus\n")
print('*'*80)
print("\n")


NB_TO_SHOW = 30
def mots_similaires():
    emb1, emb2 = embeddings['DROITE'], embeddings['GAUCHE']
    vocab1, vocab2 = emb1.get_vocab().index_to_key, emb2.get_vocab().index_to_key
    sims = {}
    ''' On compare les vecteurs des mots qui sont dans les deux vocabulaires '''
    for word in vocab1:
        if word in vocab2:
            sims[word] = cosine_similarity(emb1.get_vector(word), emb2.get_vector(word))

    ''' On trie les similarités dans l'ordre croissant et décroissant '''        
    sims_up = {k: v for k, v in sorted(sims.items(), key=lambda item: item[1], reverse=True)}
    sims_down = {k: v for k, v in sorted(sims.items(), key=lambda item: item[1])}

    ''' On affiche d'abord les mots les plus proches '''     
    i = 0
    print("*** Vecteurs proches : ", end='')
    while i < NB_TO_SHOW:
        w = list(sims_up.keys())[i]
        print(w, end=', ')
        i += 1
    print("\n")
    ''' Puis les mots les plus éloignés'''     
    print("\n*** Vecteurs éloignés : ", end='')
    i = 0
    while i < NB_TO_SHOW:
        w = list(sims_down.keys())[i]
        print(w, end=', ')
        i += 1
    
mots_similaires()

### Partie 2 : Observation de vecteurs similaires

print("\n\n\n")
print('*'*80)
print("\nPartie 2 : Observation de vecteurs similaires\n")
print('*'*80)
print("\n")


mots = ['gauche', 'droite', 'homme', 'action', 'travailleurs', 'droits', 'ecologie'] 


def top_mot_similaires(embedding, key, mots, topN=10):
    for mot in mots:
        sims = ' '.join([w for w,s in embedding.most_similar(mot, topN)])
        print('- {}\t(chez la {})\t{}'.format(mot, key, sims))

for key, embedding in embeddings.items():
    top_mot_similaires(embedding, key, mots, 10)
    print("\n")
    
### Partie 3 : Analogies
    
print("\n\n\n")
print('*'*80)
print("\nPartie 3 : Analogies\n")
print('*'*80)
print("\n")
    

def mots_analogies(pos, neg):
    for key, embedding in embeddings.items():
        most_sim = embedding.most_similar_analogy(pos, neg, topN=5) 
        print(f"Analogie chez la {key} : {pos[0]} - {neg[0]} + {pos[1]} :")
        for w_s in most_sim:
            word, sim = w_s 
            print('{}\t{}'.format(word, sim))
        print("\n")

mots_analogies(['président', 'femme'], ['homme'])
mots_analogies(['candidat', 'femme'], ['homme'])
mots_analogies(['député', 'candidats'], ['candidat'])
