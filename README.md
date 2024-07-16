# NLP

Introducción al curso e inicio de temas.

## APLICACIONES DEL NLP

-   Detección de spam
-   Texto predictivo
-   Reconocimiento de voz
-   Análisis de sentimientos
-   La automatización de NLP

## Avances recientes en NLP

-   Traducción del lenguaje
-   Generación de texto
-   Caso de uso: gpt-4

## Procesamiento de texto en la computadora

-   conversión de texto en vectores
-   métodos basados en vectores
-   Aplicaciones del mundo real

##Visión probabilística de NLP

-   Modelos de lenguaje y modelos de Markov
-   Generación de texto
-   Spinning de Artículos
-   Descifrado de códigos

## Métodos de Machine Learning para NLP

-   Detección de spam y análisis de sentimientos
-   Indexación semántica latente y modelado de temas
-   Aplicaciones del mundo real de estos métodos

## Deep Learning y Redes Neuronales para NLP

-   Redes neuronales completamente conectadas, Redes neuronales convolucionales, RN recurrentes, Arquitecturas modernas como el Transformador, Aplicaciones de estas arquitecturas en NLP

## Importancia de los vectores en el aprendizaje automático y el análisis de datos

-   Los vectores son representaciones numéricas de datos, en este caso, texto
-   Es la forma en que traducimos el texto a un formato que un modelo de aprendizaje automático puede entender y aprender

##### Ventajas de trabajar con representaciones numéricas.

-   En lugar de trabajar con un texto sin procesar, podemos convertir el texto en vectores, que son representaciones numéricas del texto

##### Utilidades especificas de los vectores en PLN

-   Detección de spam utilizando vectores
-   Organización de documentos mediante el uso de vectores.

##### Desafíos y consideraciones al convertir texto en vectores

-   No todas las representaciones vectoriales son útiles
-   El objetivo es obtener representaciones vectoriales útiles

## Bolsa de palabras (bag of words)

Técnica común en el NLP en donde cada número va a representar una palabra

##### Enfoque del procesamiento del lenguaje natural

-   La prevalencia de los enfoques de NLP que no consideran el orden de las palabras.
-   La propuesta de una representación numérica del texto que no considera el orden de las palabras: la bolsa de palabras.

###### Ejemplo

-   Ejemplo de la pérdida de información con el uso de la bolsa de palabras
-   Aclaración sobre las diferencias basadas en el orden de las palabras

        Un juego de salón == Un salón de Juego

##### Aplicación

-   Uso en modelos vectoriales y aprendizaje automático clásico.
-   Comparación con otros enfoques, como los modelos probabilísticos y el aprendizaje profundo

##### Valor y eficacia

-   Reflexión sobre las limitaciones y la precisión
-   Reconocimiento de la utilidad de la bolsa de palabras en el análisis y procesamiento del lenguaje natural

### Tokenización

-   Dividir el texto en tokens
-   Los tokens son las unidades individuales de un texto
-   Por ejemplo, las palabras son tokens en un proceso de tokenización basado en palabras

##### Comparación de la tokenización moderna y antigua

-   solía ser tan simple como separar solo por espacios
-   La tokeniazación moderna es más sofisticada

###### Diferentes perspectivas para abordar el tema

-   Basada en palabras (casa,perro,etc.)
    -   Este es más natural
    -   Más rápida de procesar
    -   Información contextual por token.
    -   Puede llegar a fallar si recibe una palabra con la cual no fue entrenado
-   Basada en caracteres (a,d,4,#)
    -   Puede manejar cualquier conjugación de palabras, e incluso palabras incompletas o mal escritas
    -   Tarda más en procesar la información por la cantidad de tokens que puede obtener
    -   Se pierde el significado semántico.
-   Basada en sub-palabras (automóvil => auto, móvil)

##### Consideraciones durante la tokenización

-   Diferentes casos de letras (tildes por ejemplo)
-   Tokenizar por palabras completas, caracteres, o sub-palabras
-   Puntuación (parte de la palabra o token separado)

## Método de Conteo

##### Definición de Documento

-   Diversos significados de "documento"
-   Ejemplos de tipos de documentos en diferentes tareas.

##### método

-   Proceso para determinar el tamaño del vocabularios "Cada palabra única" ( tupas() )
-   Proceso de creación de un vector para cada documento basado en el conteo de palabras.
-   verificación de que palabras aparecen más y menos

##### Desafíos Prácticos

-   Tokenización
-   Mapeo.

##### Relevancia del volumen de datos para el aprendizaje de modelos

-   El volumen de datos es crucial para el aprendizaje de modelos

### Stemming y Lemmatization

-   El stemming es una técnica más simple que elimina los sufijos de la palabra
-   La lemmatization es una técnica más sofisticada que utiliza las reglas del lenguaje para obtener la base o raíz de una palabra

    -   Este puede llegar a ser más tardado

    -   para la lemmatization ocupe 2 modulos extra

        ```bash
            pip install spacy
            python -m spacy download es_core_news_sm
        ```

    -   Puede ser más efectiva que el stemming, pero tambien es más costosa computacionalmente
    -   El uso de la lemmatization puede requerir el etiquetado previo.

    ##### Aplicaciones reales

    -   Asistentes virtuales y chatbots
    -   Análisis de sentimientos
    -   Motores de búsqueda
    -   Sistemas de recomendación
    -   Aplicación en publicidad online y etiquetas de redes sociales

### Ya existen Datasets que podemos usar

Hay varios en especial en la web [kaggle](https://www.kaggle.com/datasets)

## Método TF-IDF

### Similitud de vectores

visualizaremos cuales tokens se parecen más y generar sus aplicaciones

###### aplicaciones

-   Similitud de documentos
-   Spinning de artículos y SEO
-   Recomendaciones
-   Chat bots

###### Calcular la similitud de vectores

-   Distancia Euclidiana
    -   d = distancia
    -   p = algo
    -   q = algo
    -   d(p,q)^2 = (q1 - p1)^2 + (q2 - p2)^2
-   ✅ Ángulo entre vectores
    -   cos n = 1 -> máxima similitud

##### Comparación entre la similitud del coseno y la distancia euclidiana

-   En el procesamiento del lenguaje Natural, a menudo se prefiere la similitud del coseno porque toma en cuenta el ángulo entre los vectores, que puede ser más relevante en contextos donde los vectores representan textos o palabras

-   Todo esto es importante ya que entre más similares sean los vectores, más conjuntos de datos podemos obtener y generar grupos más específicos

-   Term frequency (TF) e Inverse document frequency (IDF)

##### Problemas con el método de conteo

-   No toma en cuenta la relevancia de las palabras
-   TF-IDF reduce el peso de las palabras comunes y aumenta el peso de las palabras que no se utilizan con frecuencia
    -   aquí mismo entran las stop word
        -   Palabras comunes como si, la, el, más, etc.
        -   si se dejan en el texto, pueden hacer que los algoritmos sean más precisos y rápidos, ya que hay menos palabras a procesar
    -   Ambigüedad y especificidad de las stop words dependiendo de la aplicación
        -   No todas las palabras comunes son stop words en todas las situaciones
        -   Por ejemplo, "no" podría ser una stop word en muchos contextos, pero no en el análisis de sentimientos, donde podría ser una palabra clave

### ¿Cómo funciona TF-IDF

-   TF-IDF funciona al asignar un puntaje a cada palabra en un documento en función de su frecuencia en ese documento (TF) y su frecuencia en todos los documentos (IDF)
-   Cuanto más a menudo aparece una palabra en un solo documento, pero menos a menudo en todos los documentos, mayor es su puntaje TF-IDF.
-   Las palabras que aparecen con más frecuencia en un documento pero raramente en otros documentos son más importantes.

### ¿Cómo se calcula?

-   TF-IDF se calcula multiplicando dos componentes: TF e IDF.
-   TF se calcula como el número de veces que aparece una palabra en un documento dividido por el total de palabras en ese documento
-   IDF se calcula como el logaritmo del total de documentos dividido por el número de documentos que contiene la palabra
    -   TF-IDF = TF(t,d) \* IDF (t,D)
    -   TF(t,d) = (Número de veces que el término t aparece en el documento d) / (Total de términos en el documento d)
    -   IDF(t,D) = log_e(Total de documentos en el corpus D / Número de documentos donde el término t aparece)
