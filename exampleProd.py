
import openpyxl

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from unidecode import unidecode
import spacy


def leer_columna_xlsx(nombre_archivo, nombre_columna):
    wb = openpyxl.load_workbook(nombre_archivo)
    hoja = wb.active

    columna = hoja[nombre_columna]
    valores = [celda.value for celda in columna]

    return valores



def preprocess_text(text):
    # Cargar el modelo de spaCy en español
    nlp = spacy.load('es_core_news_sm')

    # Convertir a minúsculas y quitar tildes
    text = unidecode(text.lower())

    # Remover puntuación y caracteres especiales
    text = re.sub(r'[^\w\s]', '', text)

    # Remover números
    text = re.sub(r'\d+', '', text)

    # Lemmatización utilizando spaCy
    doc = nlp(text)
    
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    
    # Remover espacios adicionales
    lemmatized_text = re.sub(r'\s+', ' ', lemmatized_text)

    # Convertir a minúsculas y quitar tildes
    lemmatized_text = unidecode(lemmatized_text.lower())

    # Remover artículos (por ejemplo, "el", "la", "los", "las")
    conectores = ['que', 'se', 'la', 'del', 'esta', 'el', 'no', 'las', 'ya', 'me', 'otros', 'el', 'yo', 'ser', 'este',
                  'a', 'al', 'ante', 'bajo', 'cabe', 'como', 'con', 'contra', 'de', 'desde', 'durante', 'en',
                  'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras',
                  'versus', 'vía', 'y', 'o', 'pero', 'aunque', 'si', 'cuando', 'porque', 'pues', 'por lo tanto',
                  'así', 'entonces', 'entonces', 'mientras', 'entretanto', 'además', 'también', 'incluso',
                  'no solo', 'sino también', 'por ejemplo', 'por otro lado', 'por lo general', 'en cambio',
                  'sin embargo', 'no obstante', 'a pesar de', 'a raíz de', 'a partir de', 'en resumen', 'en definitiva',
                  'en conclusión', 'en síntesis', 'en breve', 'en concreto', 'en general', 'por consiguiente',
                  'por ende', 'de hecho', 'en realidad', 'en efecto', 'en verdad', 'es decir', 'o sea', 'más bien',
                  'al contrario', 'en particular', 'especialmente', 'particularmente', 'fundamentalmente',
                  'principalmente', 'sustancialmente', 'probablemente', 'posiblemente', 'eventualmente',
                  'eventual', 'finalmente', 'últimamente', 'primeramente', 'previamente', 'anteriormente',
                  'posteriormente', 'consecuentemente', 'por consiguiente', 'por lo tanto', 'en consecuencia',
                  'a raíz de', 'debido a', 'gracias a', 'a causa de', 'por motivo de', 'por culpa de',
                  'con respecto a', 'en relación a', 'en cuanto a', 'en lo que respecta a', 'en lo que se refiere a',
                  'en términos de', 'en función de', 'en base a', 'a propósito de', 'respecto de', 'en virtud de',
                  'en vista de', 'en consideración a', 'en atención a', 'en comparación con', 'a diferencia de',
                  'al igual que', 'similarmente', 'por analogía', 'a semejanza de', 'en contraste con',
                  'a diferencia de', 'a pesar de', 'no obstante', 'aunque', 'mientras que', 'si bien', 'salvo que',
                  'excepto que', 'en caso de que', 'suponiendo que', 'siempre que', 'a no ser que', 'a menos que',
                  'en el supuesto de que', 'con tal de que', 'en aras de', 'con el fin de', 'con el propósito de',
                  'con el objetivo de', 'con el objeto de', 'para que', 'a fin de que', 'a efecto de que',
                  'con la intención de', 'con la finalidad de', 'a los efectos de', 'con miras a', 'a propósito de'
                  ]

    lemmatized_text = re.sub(r'\b(?:' + '|'.join(conectores) + r')\b', '', lemmatized_text)
    
    # Retornar el texto preprocesado
    return lemmatized_text


# Lectura del archivo
nombre_archivo = 'incidentes_cliente_360.xlsx'
nombre_columna = 'A'  # Letra de la columna que deseas leer
documents = leer_columna_xlsx(nombre_archivo, nombre_columna)

# Preprocesamiento de texto y creación de la matriz de características
preprocessed_documents = [preprocess_text(doc).split() for doc in documents]
vectorizer = CountVectorizer()
matrix_de_caracteristicas = vectorizer.fit_transform([' '.join(doc) for doc in preprocessed_documents])

# Entrenamiento del modelo LDA
num_temas = 6  # Número de temas a identificar
max_iter = 1000
lda_model = LatentDirichletAllocation(n_components=num_temas, random_state=10, max_iter=max_iter)

lda_model.fit(matrix_de_caracteristicas)

# Obtener las características más importantes para cada tema
num_caracteristicas = 20  # Número de características más importantes para mostrar
temas_caracteristicas = lda_model.components_.argsort(axis=1)[:, :-num_caracteristicas-1:-1]

# Mostrar los temas y sus características más importantes
features = vectorizer.get_feature_names_out()
for i, tema in enumerate(temas_caracteristicas):
    print(f"Temas {i+1}:")
    for indice in tema:
        caracteristica = features[indice]
        porcentaje = lda_model.components_[i][indice]
        print(f"{caracteristica}: {porcentaje}")
    print()
