import gensim
import re
from gensim import corpora
import spacy


# Función para preprocesar el texto
def preprocess_text(text):
    # Cargar el modelo de spaCy en español
    nlp = spacy.load('es_core_news_sm')
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Remover artículos (por ejemplo, "el", "la", "los", "las")
    conectores = ['que', 'se', 'la', 'del', 'esta', 'el', 'no', 'las', 'ya', 'me', 'otros',
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

    
    text = re.sub(r'\b(?:' + '|'.join(conectores) + r')\b', '', text)
    
    # Remover puntuación y caracteres especiales
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remover números
    text = re.sub(r'\d+', '', text)
    
    # Lemmatización utilizando spaCy
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    
    # Remover espacios adicionales
    lemmatized_text = re.sub(r'\s+', ' ', lemmatized_text)
    
    # Retornar el texto preprocesado
    print("text: ", text)
    print("lematized: ",lemmatized_text)
    return lemmatized_text


# Datos de ejemplo
documents = [
    "En primavera, las flores florecen y el aire se llena de aromas.",
    "El verano trae consigo días soleados y diversión en la playa.",
    "Durante el otoño, los árboles se cubren de tonos cálidos y las hojas caen al suelo.",
    "En invierno, la nieve cubre el paisaje y se disfrutan las bebidas calientes junto al fuego.",
    "La primavera es una época de renacimiento y crecimiento.",
    "El verano es perfecto para disfrutar del aire libre y los deportes acuáticos.",
    "El otoño nos regala paisajes pintorescos y deliciosas calabazas.",
    "El invierno trae consigo la magia de las fiestas y la alegría de la nieve.",
    "Disfrutar de un picnic en primavera es una delicia para los sentidos.",
    "Las vacaciones de verano son ideales para viajar y explorar nuevos destinos.",
    "Los colores cálidos del otoño crean una atmósfera acogedora y nostálgica.",
    "En invierno, es divertido construir muñecos de nieve y deslizarse por las colinas.",
    "La primavera es la estación perfecta para observar la naturaleza en pleno esplendor.",
    "Los días largos y soleados del verano invitan a disfrutar de actividades al aire libre.",
    "El otoño nos regala el placer de caminar entre las hojas secas y saborear las manzanas.",
    "En invierno, se puede patinar sobre hielo y disfrutar de chocolate caliente.",
    "Los jardines florecen en primavera, llenando el entorno de color y fragancia.",
    "Durante el verano, las playas se convierten en destinos de ensueño para relajarse.",
    "El otoño nos regala hermosos atardeceres y la cosecha de frutas deliciosa.",
    "En invierno, es mágico ver caer la nieve y acurrucarse junto a la chimenea.",
    "La primavera es la estación ideal para realizar actividades al aire libre en familia.",
    "El verano nos brinda la oportunidad de disfrutar del sol y refrescarnos en la piscina.",
    "El otoño nos deleita con el crujir de las hojas bajo nuestros pies y el olor a tierra mojada.",
    "En invierno, podemos deleitarnos con la belleza de los copos de nieve y el aroma a canela.",
    "Los días soleados de primavera nos invitan a dar paseos y disfrutar de la naturaleza.",
    "Durante el verano, podemos disfrutar de deliciosos helados y barbacoas al aire libre.",
    "El otoño nos permite disfrutar de la calidez de una taza de té y la contemplación de paisajes otoñales.",
    "En invierno, podemos disfrutar de la emoción de los deportes de invierno como el esquí y el snowboard.",
    "La primavera es la estación en la que los animales despiertan de su letargo invernal.",
    "En verano, los días son más largos y las noches son perfectas para observar las estrellas.",
    "El otoño nos brinda la oportunidad de recolectar castañas y disfrutar de tardes acogedoras en casa.",
    "En invierno, podemos disfrutar de la compañía de nuestros seres queridos durante las festividades.",
    "La primavera es el momento ideal para renovar nuestro jardín y plantar nuevas flores y plantas.",
    "Durante el verano, podemos disfrutar de refrescantes baños en el mar y hacer castillos de arena.",
    "El otoño nos invita a degustar platos reconfortantes como sopas y guisos calientes.",
    "En invierno, podemos disfrutar de las luces brillantes y la alegría de la temporada navideña.",
    "La primavera es una época en la que los días se vuelven más cálidos y agradables.",
    "El verano nos permite disfrutar de deliciosas frutas frescas y sabrosos helados.",
    "El otoño nos regala hermosos paisajes cubiertos de hojas doradas y rojizas.",
    "En invierno, podemos disfrutar de la belleza de los copos de nieve y las veladas junto al fuego.",
    "La primavera es una estación de esperanza y nuevos comienzos.",
    "Durante el verano, podemos disfrutar de largas tardes al sol y refrescarnos en piscinas o lagos.",
    "El otoño nos invita a dar paseos entre árboles que cambian de color y a saborear calabazas y boniatos.",
    "En invierno, podemos disfrutar de la emoción de los deportes de nieve y las fiestas navideñas.",
    "La primavera es el momento perfecto para disfrutar de picnics en el campo y admirar las flores silvestres.",
    "Durante el verano, podemos disfrutar de vacaciones en la playa y barbacoas en el jardín.",
    "El otoño nos trae una sensación de nostalgia y la oportunidad de abrigarnos con suéteres cálidos.",
    "En invierno, podemos disfrutar de la magia de las luces navideñas y los regalos bajo el árbol.",
    "La primavera nos llena de energía y nos inspira a realizar nuevos proyectos y metas.",
    "Durante el verano, podemos disfrutar de festivales al aire libre y actividades recreativas.",
    "El otoño nos invita a disfrutar de tardes tranquilas leyendo un buen libro junto a la chimenea.",
    "En invierno, podemos disfrutar de deliciosas comidas caseras y el calor de nuestros hogares.",
    "La primavera es una estación en la que podemos disfrutar del canto de los pájaros y el aroma de las flores.",
    "Durante el verano, podemos disfrutar de emocionantes deportes acuáticos y relajarnos bajo el sol.",
    "El otoño nos brinda la oportunidad de disfrutar de la belleza de los paisajes otoñales y los sabores de la temporada.",
    "En invierno, podemos disfrutar de la emoción de los mercados navideños y la alegría de regalar a nuestros seres queridos.",
    "La primavera es el momento ideal para limpiar y organizar nuestra casa y dar la bienvenida a la renovación.",
    "Durante el verano, podemos disfrutar de paseos en bicicleta y picnics en parques verdes.",
    "El otoño nos invita a disfrutar de la calidez de las bebidas especiadas y las mantas suaves.",
    "En invierno, podemos disfrutar de películas acogedoras y pasar tiempo de calidad con nuestra familia.",
    "La primavera es una estación en la que podemos disfrutar de la belleza de los campos llenos de flores.",
    "Durante el verano, podemos disfrutar de noches estrelladas y divertidas fiestas al aire libre.",
    "El otoño nos brinda la oportunidad de disfrutar de la cosecha de frutas y verduras frescas.",
    "En invierno, podemos disfrutar de la emoción de los deportes de invierno y la magia de la Navidad.",
    "La primavera es una época en la que podemos disfrutar de largos paseos por la naturaleza y respirar el aire fresco.",
    "Durante el verano, podemos disfrutar de deliciosos helados y refrescantes bebidas frías.",
    "El otoño nos invita a disfrutar de tardes acogedoras en casa con una taza de té caliente.",
    "En invierno, podemos disfrutar de la belleza de los paisajes nevados y los adornos festivos.",
    "La primavera nos inspira a renovar nuestro guardarropa y llenarlo de colores alegres.",
    "Durante el verano, podemos disfrutar de viajes emocionantes y aventuras al aire libre.",
    "El otoño nos brinda la oportunidad de disfrutar de hogueras y deliciosas comidas reconfortantes.",
    "En invierno, podemos disfrutar de la compañía de nuestros seres queridos y celebrar la paz y el amor.",
    "La primavera es una estación en la que podemos disfrutar de picnics en parques llenos de flores.",
    "Durante el verano, podemos disfrutar de refrescantes baños en piscinas y lagos cristalinos.",
    "El otoño nos invita a disfrutar de largas caminatas en el bosque y recolectar hojas secas.",
    "En invierno, podemos disfrutar de la magia de las luces de Navidad y los sabores tradicionales.",
    "La primavera nos llena de esperanza y nos invita a disfrutar de días más cálidos y soleados.",
    "Durante el verano, podemos disfrutar de días de playa y divertidos juegos en el agua.",
    "El otoño nos brinda la oportunidad de disfrutar de la belleza de los árboles cubiertos de hojas doradas.",
    "En invierno, podemos disfrutar de la alegría de regalar y recibir regalos durante las fiestas."
]

# Preprocesamiento de texto y creación del corpus
preprocessed_documents = [preprocess_text(doc).split() for doc in documents]


dictionary = corpora.Dictionary(preprocessed_documents)



corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]


# Entrenamiento del modelo LDA
num_topics = 4
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=200)

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

