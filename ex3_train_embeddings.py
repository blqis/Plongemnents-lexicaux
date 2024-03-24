from embeddings import Embeddings

droite_path = './data_embeddings/DROITE.txt'
gauche_path = './data_embeddings/GAUCHE.txt'

model_dr_path = './models/DROITE.W2Vmodel'
model_ga_path = './models/GAUCHE.W2Vmodel'



embeddings_dr = Embeddings(droite_path, model_dr_path)
embeddings_dr.learn()

print(f"Model {model_dr_path} saved.")

embeddings_ga = Embeddings(gauche_path, model_ga_path)
embeddings_ga.learn()

print(f"Model {model_ga_path} saved.")