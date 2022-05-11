# Tirocinio Triennale Univpm 2020-2021
Progettazione e sviluppo di un algoritmo basato su Deep Learning per la classificazione di immagini RGB nell’ambito della sicurezza stradale 

Design and Development of an algorithm based on Deep Learning to classify RGB images in the context of road safety

# Obiettivo Progettuale
L’obiettivo di questo progetto è quello di implementare e addestrare una Rete Neurale in grado di rilevare i movimenti del guidatore. In modo particolare focalizziamo la nostra nostra attenzione sull’apertura e la chiusura degli occhi. Per poter ridurre al minimo gli errori commessi, analizziamo anche i falsi positivi e i falsi negativi.
Questo lavoro è stato diviso in due parti: la prima parte relativa a Facemesh, la seconda riguarda l’addestramento della Rete Neurale tramite un set di immagini non molto ampio.


## Strumenti Utilizzati
- Media Pipe FaceMesh
- MobileNetV2
- MRL Dataset (dataset già preimpostato, utilizzato per l'addestramento della rete neurale)

## Librerie
- OpenCV : libreria utilizzata nell'ambito della computer vision e del machine learning, serve principalmente per la visualizzazione di immagini
- Keras : utilizzato per l'addestramento di Reti Neurali, funge da interfaccia semplice (API) per l’accesso e
programmazione a diversi Framework di apprendimento automatico.
Tra questi Framework supporta le librerie TensorFlow, Microsoft Cognitive Toolkit (in precedenza CNTK) e Theano.

## Linguaggi Utilizzati
Il principale linguaggio di programmazione utilizzato è stato Python

## Editor di sviluppo
- VS Code : Ide
- Google Colaboratory : utilizzato per la parte di addestramento della rete neurale

## Sviluppo del Progetto
- Addestramento iniziale della rete neurale appoggiandosi al Dataset MRL Eye 
- Addestramento Rete Neurale : Test Set e Validation Set, Data Augmentation, Implementazione MobileNetV2, Feature Extraction
- Validation (Fine Tuning, Evaluation and Prediction)
- Test Locale


