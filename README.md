# Plongemnents lexicaux

## Utilisation

Le fichier `ex4_use_embeddings.py` regroupe les questions des exercices 3 et 4 et les affiches (formatées) sur le terminal à l'exécution. Le fichier `ex5_plot,py` quant à lui affiche trois graphiques : deux basés sur un même sous-corpus de gauche (pour vérifier l'inéquivalence), ainsi qu'un de droite.

## Exercice 2

### Biblipothèque et modèle utilisé
La bibliothèque utilisée pour procéder à l'apprentissage des embeddings est Gensim.

Le modèle spécifique utilisé est Word2Vec, classe présente dans Gensim.

### Limites dans le processus général de construction des embeddings :
- corpus de départ : embeddings construits à partir de ces données seulement, difficilement réutilisables dans d'autres contextes au vu de leur spécificité
	
### Améliorations possibles :
- lemmatisation ou "subwords embeddings" pour gérer la flexion complexe du français
- modifier les attributs du modèle Word2vec pour étendre l'apprentissage (taille de vecteur, fenêtre plus grande, plus de workers...)

## Exercice 3

### Vecteurs du mot "patrie"

Les deux vecteurs ne sont pas égaux, étant donné que l'apprentissage de chaque embeddings a été fait sur une sous-partie différente du corpus, et donc un contexte de mot différent.

Quant à la taille des vecteurs, elle est de 100, vu que la classe Word2vec a pour attribut `vector_size: int = 100` par défaut. Il est donc tout à fait possible de changer cette valeur au sein de notre fonction `embeddings.learn()`, comme on a pu le faire avec le paramètre `sentences = sentences`, ou tout simplement à l'appel de celle-ci.

## Exercice 4

### Étude des vecteurs proches et éloignés

Les mots aux vecteurs les plus proches sont en général des mots utilisés dans des contextes spécifiques, on peut penser à des évenements précis ou des arguments similaires aux deux camps.

Cette analyse est assez intéressante dans le sens où l'on peut assez facilement deviner les similarité entre ces parti politiques "opposés", et au contraire les points de divergence qui les caractérise : je pense notamment aux mots 'droit', 'écologie', 'travailleurs', qui sont des thèmes dont la vision d'un parti est diamétralement opposé à l'autre, et cela se reflète dans les résultats.

Quant aux mots vide, il y en a étonamment peu : hormis 2 ou 3 ("nous", "ce", "jusqu"), on se retrouve avec un apprentissage assez pertinent et des résultats sur lesquels on peut baser des hypothèses et un certain raisonnement.

### Observation de vecteurs similaires

Les rapprochements de mots affichés sont assez pertinents et on pourrrait même dire qu'ils sont assez représentatifs : ce que "pense" la gauche de son propre parti, de la droite, et vice versa. Comme énoncé précedemment, on le devine si bien à travers ces résultats (assez variables selon le corpus observé) que c'en est presque une caricature des discours politiques et l'idéologie de chaque parti.

## Exercice 5 

### Analogies

Dans le cas `président - homme + femme`, à droite le mot "présidente" arrive en troisième place, tandis qu'il n'apparaît pas du tout à gauche, alors que "députée" est en seconde place !

Pour la droite, cela s'explique sûrement par les échecs, euh, campagnes répétées d'une certaine candidate d'un parti néo nazi, euh, du RN.

Pour la gauche, disons que les femmes ne sont a priori pas les candidates phares à la présidentielle...

Mais l'on retrouve bien, pour les deux parti, le mot "candidate" à la première place pour l'analogie `candidat - homme + femme`. ¨Députée" suit de très près cependant...

Enfin, dans le cas de `député - candidat + candidats`, on retrouve à droite "élus", "ministres", "présidents", et à gauche plutôt "suffrages" et "députés", dans cet ordre.

### Questions

Deux apprentissages d'embeddings ne garantissent pas des vecteurs strictement équivalents. Le processus inclut certaines tâches à l'initialisation aléatoire, ce qui affecte le résultat final. Les paramètres d'apprentissage influent également.

Quant aux graphiques générés sur un même corpus, on observe des regroupements (ou clusters) similaires, mais la structure générale est légèrement différente, dû aux facteurs aléatoires susmentionnés. La distribution n'est pas la même.