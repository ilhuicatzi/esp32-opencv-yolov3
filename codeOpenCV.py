
import cv2
import numpy as np
import urllib.request

# URL de la cámara web
url = 'http://10.42.0.113/cam-mid.jpg'

# Inicialización de la captura de video desde la URL
cap = cv2.VideoCapture(url)

# Parámetros de detección de objetos
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Archivo de nombres de clases
classesfile = 'coco.names'

# Lectura de los nombres de las clases desde el archivo
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('
').split('
')

# Carga del modelo YOLO
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)

# Configuración del backend y el objetivo preferidos para la red neuronal
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Función para encontrar objetos en la imagen
def findObject(outputs, im):
    # Extracción de dimensiones de la imagen
    hT, wT, cT = im.shape
    
    # Listas y conjunto para almacenar información de detección
    bbox = []
    classIds = []
    confs = []
    found_classes = set()

    # Iteración sobre las salidas de detección
    for output in outputs:
        for det in output:
            # Extracción de puntuaciones de confianza
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            
            # Verificación de confianza
            if confidence > confThreshold:
                # Cálculo de coordenadas del cuadro delimitador
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                
                # Almacenamiento de información de detección
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                found_classes.add(classNames[classId])

    # Aplicación de NMS para eliminar cuadros delimitadores redundantes
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Iteración sobre los índices de los cuadros delimitadores seleccionados
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        className = classNames[classIds[i]]
        found_classes.add(className)

        # Dibujado del cuadro delimitador y el texto en la imagen
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(im, f'{className.upper()} {int(confs[i]*100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Impresión de las clases encontradas
    print("Objetos encontrados:", found_classes)

# Bucle principal para capturar y procesar continuamente las imágenes
while True:
    # Obtención y decodificación de la imagen desde la URL
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    # Procesamiento de la imagen para la detección de objetos
    blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputNames = [layernames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)

    # Llamada a la función findObject para encontrar y dibujar los objetos
    findObject(outputs, im)

    # Visualización de la imagen con los objetos detectados
    cv2.imshow('IMage', im)
    cv2.waitKey(1)
