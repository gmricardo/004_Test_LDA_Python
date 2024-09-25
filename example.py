import gensim
import re
import spacy
from gensim import corpora



# Función para preprocesar el texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Remover artículos (por ejemplo, "el", "la", "los", "las")
    text = re.sub(r'\b(?:el|la|los|las|son|un|en|de|es|está|están|un|una|unos|unas|y|o|para|con|por)\b', '', text)
    # Remover puntuación y caracteres especiales
    text = re.sub(r'[^\w\s]', '', text)
    # Remover números
    text = re.sub(r'\d+', '', text)
    # Remover espacios adicionales
    text = re.sub(r'\s+', ' ', text)
    # Retornar el texto preprocesado
    return text


# Datos de ejemplo
documents = [
    "el sol es brillante para",
    "las flores son hermosas",
    "el cielo está despejado",
    "las aves cantan en los árboles",
    "disfruto de un día soleado"
]

# Preprocesamiento de texto y creación del corpus
preprocessed_documents = [preprocess_text(doc).split() for doc in documents]
# Separa cada una de las palabras quitando los conectores y demas palabras que no den significado.
#print(preprocessed_documents)
#[['sol', 'brillante'], ['flores', 'hermosas'], ['cielo', 'despejado'], ['aves', 'cantan', 'árboles'], ['disfruto', 'día', 'soleado']]

dictionary = corpora.Dictionary(preprocessed_documents)
# Se crea un diccionario con todas las palabras de los documentos
#print(dictionary)
# 12 unique tokens: ['brillante', 'sol', 'flores', 'hermosas', 'cielo']...


corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]
#Cuenta cuantas veces aparece una palabra en un documento o linea
#print("\nCorpus:")
#print(corpus)
#[[(0, 1), (1, 1)], [(2, 1), (3, 1)], [(4, 1), (5, 1)], [(6, 1), (7, 1), (8, 1)], [(9, 1), (10, 1), (11, 1)]]



# Entrenamiento del modelo LDA
num_topics = 2
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=100)

# Obtención de los tópicos y palabras clave asociadas
topics = lda_model.print_topics(num_topics=num_topics)
for topic in topics:
    print(topic)

# Analisis de identidades
# Cargar el modelo de lenguaje en español
nlp = spacy.load('es_core_news_sm')

def entities_text(text):
    # Procesar el texto con SpaCy
    doc = nlp(text)

    # Extraer las entidades nombradas
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Imprimir las entidades encontradas
    for entity, label in entities:
        print(entity, label)

# Preprocesamiento de texto y creación del corpus
entidades = [entities_text(doc) for doc in documents]


