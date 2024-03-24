from embeddings import Embeddings

droite_path = './data_embeddings/DROITE.txt'
gauche_path = './data_embeddings/GAUCHE.txt'

model_dr_path = './models/DROITE.W2Vmodel'
model_ga_path = './models/GAUCHE.W2Vmodel'

embeddings_dr = Embeddings(droite_path, model_dr_path)
embeddings_dr.load()

embeddings_ga = Embeddings(gauche_path, model_ga_path)
embeddings_ga.load()

models = {"DROITE": embeddings_dr, "GAUCHE": embeddings_ga}

def ex3_test_embeddings(models):    
    for key, model in models.items():
        print(f"Modèle {key} chargé.")
        print(f"Taille du modèle en nombre de mots : {len(model.get_vocab())}")
        print(f"Vecteur du mot 'patrie' : {model.get_vector('patrie')}")
        print("\n")
