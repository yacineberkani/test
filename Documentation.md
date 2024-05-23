# Étude Comparative des Documents avec RAG
 -  Étude Comparative des Documents Utilisant la Technique de RAG avec Knowledge Graph(graphiques de connaissances) et Llama-Index [Medium_RAG_with_KnowledgeGraph](https://medium.aiplanet.com/implement-rag-with-knowledge-graph-and-llama-index-6a3370e93cdd)

**Introduction**

L'analyse comparative automatique de documents représente l'une des tâches les plus complexes de machine learning , faisant appel à une multitude de techniques variées telles que la reconnaissance d'entités nommées (NER) vers l'extraction de relations (RE). Avec l'avancée de l'IA générative, notamment avec le modèle de langage LLM (Large Language Models), cette approche devient plus accessible en exploitant des techniques telles que le RAG (Génération Augmentée par Récupération), qui améliore les performances des LLM

**RAG (Génération Augmentée par Récupération)**
La RAG consiste à améliorer les performances des LLM en récupérant des morceaux de documents pertinents à partir d'une base de connaissances externe telles que :

**Base de données vectorielles**
![image1](https://github.com/yacineberkani/test/blob/main/img1.png)
Une base de données vectorielles consiste en une compilation de vecteurs à haute dimension représentant diverses entités ou concepts, comme des mots, des phrases dans un documents. Son utilisation principale réside dans l'évaluation de la similitude ou des liens entre ces entités, basée sur leurs représentations vectorielles.


**Graphe de Connaissances (Knowledge Graph)**
![image2](https://github.com/yacineberkani/test/blob/main/igm2.png)
Un graphe de connaissances est une structure composée de nœuds et d'arêtes, symbolisant respectivement des entités ou concepts ainsi que leurs relations, telles que des faits ou des propriétés.

Son utilisation principale consiste à extraire ou déduire des informations factuelles sur diverses entités ou concepts, en se basant sur les attributs associés aux nœuds et aux arêtes

Ces informations récupérées sont ensuite utilisées pour enrichir le contexte de génération du modèle, réduisant ainsi les risques de produire les hallucination (contenus incorrects)

** Pourquoi les graphes de connaissances plutôt que les bases de données vectorielles ? 

 - Représentation des relations complexes
 - Capacités d'analyse sémantique
 - Raisonnement avancé
 - Répondre à des requêtes complexes

Notre étude a choisi d'utiliser les graphes de connaissances pour plusieurs raisons.

Ces derniers permettent de capturer les relations complexes entre les entités et offrent des capacités d'analyse sémantique et de raisonnement avancé. Ils sont capables de répondre à des requêtes complexes basées sur des opérateurs logiques et de réaliser une découverte de connaissances approfondie.

Contrairement aux bases de données vectorielles qui se concentrent sur la similitude entre les vecteurs

**Technologies Employées**

 - **1. LlamaIndex:**  est un framework d'orchestration qui simplifie l'intégration de données privées avec des données publiques pour créer des applications à l'aide de grands modèles linguistiques (LLM). Il fournit des outils d'ingestion, d'indexation et d'interrogation de données, ce qui en fait une solution polyvalente pour les besoins d'IA générative.
[llamaindex](https://ts.llamaindex.ai/fr/#:~:text=ce%20que%20LlamaIndex.-,TS%3F,ou%20sp%C3%A9cifiques%20%C3%A0%20un%20domaine)

![image3](https://github.com/yacineberkani/test/blob/main/Capture%20d%E2%80%99e%CC%81cran%202024-05-11%20a%CC%80%2016.31.32.png)

- **2. thenlper/gte-large:**  le modèle d'intégration est requis pour convertir le texte en représentation numérique d'une information pour le texte fourni. La représentation capture la signification sémantique de ce qui est intégré, ce qui la rend robuste pour de nombreuses applications industrielles [modèle d'intégration](https://huggingface.co/thenlper/gte-large)
- **3. Llama-3-8B (LLM):**  Le modèle Large Language est requis pour générer une réponse basée sur la question et le contexte fournis. Ici, nous avons utilisé le modèle Llama 3  [Meta-llama3](https://github.com/meta-llama/llama3)

 **Implémentation du code étape par étape :**
Création d'un graphe de connaissances à l'aide de Llama Index .

  - 1. Nous allons parcourir un fichier .pdf qui pourrait être transformé en un graphe de connaissances bien structuré
  - 2. Stockez les intégrations dans un référentiel de données graphiques.
  - 3. Récupérer le contexte pertinent correspondant à la requête de l'utilisateur
  - 4. Fournir la réponse au LLM pour générer une réponse
**Installer les dépendances requises**
```Python
!pip install llama_index pyvis Ipython langchain pypdf
!pip install llama-index-llms-huggingface
!pip install llama-index-embeddings-langchain
```
**Mettre en place le LLM**

Nous avons utilisé les points de terminaison de l'API d'inférence Huggingfac
```Python
# Définition de la clé d'API Huggingface
HF_TOKEN = "*********"  # Clé d'API Huggingface

# Création de l'API Huggingface pour l'inférence

llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",  # Nom du modèle utilisé
    token=HF_TOKEN  # Clé d'API pour l'authentification
)
```
**Configurer le modèle d'intégration**
# Création du modèle d'intégration
```Python
embed_model = LangchainEmbedding(
    HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,  # Clé d'API Huggingface pour l'authentification
        model_name="thenlper/gte-large"  # Nom du modèle pour les intégrations
    )
)
```
Charger les données. Ici, j'ai utilisé deux fichiers PDF pour faire une étude comparative.

- [PDF1](https://github.com/yacineberkani/test/blob/main/Rapport_Projet.pdf)
- [PDF2](https://github.com/yacineberkani/test/blob/main/ProjetNLP_BERKANI-Yacine_ELBIDI-Louai.pdf)


# Création d'un lecteur de répertoire simple pour charger les données depuis "/content"
doc = SimpleDirectoryReader(
    input_dir="/content",  # Répertoire d'entrée pour les fichiers
    required_exts=[".pdf", ".docx"]  # Extensions de fichiers requises
)
```Python
# Chargement des données à partir du répertoire spécifié
documents = doc.load_data()

# Affichage du nombre de page total des documents chargés
print(len(documents))

# Affichage des documents chargés
print(documents)
```

**Construire l'index du Graphe de Connaissances (Knowledge Graph)**
```Python
# Configurer le contexte de service (paramètre global de LLM)
Settings.llm = llm  # Définition du modèle de langage pour le contexte de service
Settings.chunk_size = 512  # Définition de la taille de chunk pour le traitement des données

# Configurer le contexte de stockage
graph_store = SimpleGraphStore()  # Création d'une instance de stockage de graphique simple
storage_context = StorageContext.from_defaults(graph_store=graph_store)  # Configuration du contexte de stockage par défaut

# Construire le Knowledge Graph Index
index = KnowledgeGraphIndex.from_documents(
    documents=documents,  # Les documents à indexer
    max_triplets_per_chunk=3,  # Le nombre de triplets de relation traités par bloc de données
    storage_context=storage_context,  # Contexte de stockage pour l'index
    embed_model=embed_model,  # Modèle d'intégration pour la représentation numérique
    include_embeddings=True  # Inclure les intégrations dans l'index
)
```

**Triplet dans un Graphe de Connaissances (Knowledge Graph)**

Un triplet est une unité de données de base dans le graphique. Il se compose de trois parties :

 - Sujet : le nœud auquel le triplet est associé.

 - Objet : le nœud vers lequel la relation pointe.

 - Prédicat : la relation qui relie le sujet à l'objet.

Pour illustrer cela avec un exemple plus concret, considérons le triplet suivant :

Sujet : "Paris"

Prédicat : "est la capitale de"

Objet : "France"

Ce triplet représente la relation où ("Paris")--[est la capitale de] ---> ("France").

**Knowledge Graph (Graphe de Connaissances) créé à partir du document.**
```Python
from pyvis.network import Network
from IPython.display import display
g = index.get_networkx_graph()
net = Network(notebook=True,cdn_resources="in_line",directed=True)
net.from_nx(g)
net.show("graph.html")
net.save_graph("Knowledge_graph.html")
#
import IPython
IPython.display.HTML(filename="/content/Knowledge_graph.html")
```
![Knowledge_graph](https://github.com/yacineberkani/test/blob/main/Capture%20d%E2%80%99e%CC%81cran%202024-05-23%20a%CC%80%2002.10.35.png)

## EXP de la requête 
```Python
# Définition de la requête
query = "comparer les deux documents ? "

# Configuration du moteur de requêtes
query_engine = index.as_query_engine(
    include_text=True,  # Inclure le texte dans les résultats de la requête
    response_mode="tree_summarize",  # Mode de réponse pour résumer les résultats
    embedding_mode="hybrid",  # Mode d'intégration pour la comparaison de similarité
    similarity_top_k=5,  # Nombre maximum de réponses à retourner
)

# Modèle de message pour la requête
message_template = f"""Veuillez vérifier si les éléments de contexte suivants contiennent une mention des mots-clés fournis dans la question. Si ce n'est pas le cas, ne connaissez pas la réponse, indiquez simplement que vous ne savez pas. Arrêtez-vous là. S'il vous plaît, ne faites pas d'effort pour inventer une réponse.

Question : {query}
Please provide a complete response in French.
Réponse utile :
"""

# Requête au moteur de recherche
response = query_engine.query(message_template)

# Affichage de la réponse
print(response.response.split("<|assistant|>")[-1].strip())

=================================Output==============================
Les deux documents sont des rapports de projet, mais ils ont des thèmes et des objectifs différents. Le Rapport Projet est un rapport de projet de détection de tumeurs cérébrales par IRM, tandis que le Projet NLP est un rapport de projet de classification de tweets selon leur intensité sentimentale. Les deux rapports utilisent des modèles de machine learning, mais les modèles et les techniques utilisées sont différentes. Le Rapport Projet compare les performances de trois modèles (Random Forest, VGG16 et CNN) pour la détection de tumeurs cérébrales, tandis que le Projet NLP utilise un modèle de transfer learning pour la classification de tweets. Les deux rapports ont des conclusions différentes, mais ils montrent tous deux l'efficacité des modèles de machine learning pour résoudre des problèmes spécifiques. En résumé, les deux documents sont des rapports de projet distincts qui abordent des thèmes et des objectifs différents
```





