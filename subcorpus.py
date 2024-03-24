### Exercice 1 : Constitution des sous-corpus

import io

def read_file(file_path):
    inFile = io.open(file_path, mode='r', encoding='utf-8') 
    lines = inFile.readlines() # lines : liste des lignes 
    inFile.close()
    return lines

def write_file(file_path, s):
    outFile = io.open(file_path, mode='w', encoding='utf-8') 
    outFile.write(s) # Ecriture de s dans le fichier
    outFile.close()


corpus_path = './data_embeddings/HYPERBASE_Droite_VS_Gauche.txt'
corpus = read_file(corpus_path)

## Constitution des sous-corpus

def extraire_souscorpus(corpus):
    droite, gauche = '', ''
    in_droite, in_gauche = False, False
    for line in corpus:
        if line.startswith("****"):
            if "*parti_droite" in line:
                in_droite = True
                in_gauche = False
            elif "*parti_gauche" in line:
                in_gauche = True
                in_droite = False
        if line.startswith("; _") or line.startswith("****"):
            continue
        if in_droite:
            droite += line
        elif in_gauche:
            gauche += line
    return droite, gauche

droite, gauche = extraire_souscorpus(corpus)

## Sauvegarde des deux sous-corpus

path_droite = './data_embeddings/DROITE.txt'
path_gauche = './data_embeddings/GAUCHE.txt'

write_file(path_droite, droite)
write_file(path_gauche, gauche)


