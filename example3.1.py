# Importa las librerías necesarias
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Carga el modelo LDA entrenado
lda_model = joblib.load('modelo_lda.pkl')

# Carga el vectorizador utilizado durante el entrenamiento del modelo
vectorizer = joblib.load('vectorizador.pkl')

# Define la función para clasificar el tema de un texto de entrada
def clasificar_tema(texto):
    # Transforma el texto de entrada en una matriz de términos
    matriz_texto = vectorizer.transform([texto])

    # Obtiene las probabilidades de los temas para el texto de entrada
    probabilidades = lda_model.transform(matriz_texto)

    # Obtiene el índice del tema con mayor probabilidad
    indice_tema = probabilidades.argmax()

    # Devuelve el tema correspondiente al índice
    temas = ['Verano', 'Otoño', 'Primavera', 'Invierno']
    tema_predicho = temas[indice_tema]

    return tema_predicho

# Ejemplo de uso
texto_entrada = "Algo con mucha vida y color"
tema_predicho = clasificar_tema(texto_entrada)
print("El tema predicho para el texto es:", tema_predicho)