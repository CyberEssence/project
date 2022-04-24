import cv2
import numpy as np
import os
import urllib.request

if os.path.exists('yolov3.weights') == False:
    destination = 'yolov3.weights'
    url = 'https://pjreddie.com/media/files/yolov3.weights'
    urllib.request.urlretrieve(url, destination)


net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

path = input('Введите путь к видео: ')
class_ = input('Введите название класса объекта: ')

cap = cv2.VideoCapture(path)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while (cap.isOpened()):
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(class_)
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            crop_img = img[y:y + h, x:x + w]
            cv2.imshow(path, crop_img)
            

    if cv2.waitKey(33) & 0xFF == ord('q'): 
        break 


cap.release()
#out.release()
cv2.destroyAllWindows()