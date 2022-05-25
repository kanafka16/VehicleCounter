import cv2
import csv
import numpy as np
from tracker import *

# Ініціалізація трекера
tracker = EuclideanDistTracker()

# Ініціалізація відеооб'єкта
cap = cv2.VideoCapture('video2.mp4')
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
inputSize = 320

# Поріг точності виявлення
confThreshold = 0.2
nmsThreshold = 0.2

fontColor = (0, 0, 255)
fontSize = 0.5
fontThickness = 2

# Розташування лінії
middleLinePosition = 100
upLinePosition = middleLinePosition - 15
downLinePosition = middleLinePosition + 15

# Збереження імен Coco в масив
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

# Індекси потрібних нам класів
requiredClassIndex = [2, 3, 5, 7]
detectedClassNames = []

# Ініціалізація файлів моделей
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# Конфігурація мережевої моделі
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Конфігурація мережевого бекенду
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Визначення випадкового кольору для кожного з класів
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Знаходження центру квадрата
def findCenter(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return cx, cy


# Ініціалізація списків для зберігання даних про кількість транспортних засобів
tempUpList = []
tempDownList = []
upList = [0, 0, 0, 0]
downList = [0, 0, 0, 0]


# Лічба транспортних засобів
def countVehicle(boxId, img):
    x, y, w, h, id, index = boxId

    # Шукаємо центр квадрата
    center = findCenter(x, y, w, h)
    ix, iy = center

    # Шукаємо поточну позицію транспортного засобу
    if (iy > upLinePosition) and (iy < middleLinePosition):
        if id not in tempUpList:
            tempUpList.append(id)

    elif downLinePosition > iy > middleLinePosition:
        if id not in tempDownList:
            tempDownList.append(id)

    elif iy < upLinePosition:
        if id in tempDownList:
            tempDownList.remove(id)
            upList[index] = upList[index] + 1

    elif iy > downLinePosition:
        if id in tempUpList:
            tempUpList.remove(id)
            downList[index] = downList[index] + 1

    # Креслимо круг у центрі квадрата
    cv2.circle(img, center, 2, (0, 0, 255), -1)


# Пошук виявлених об'єктів із мережевого виводу
def postProcess(outputs, img):
    global detectedClassNames

    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidenceScores = []
    detection = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in requiredClassIndex:
                if confidence > confThreshold:
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)

                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidenceScores.append(float(confidence))

    # Застосування техніки "немаксимального придушення"
    indices = cv2.dnn.NMSBoxes(boxes, confidenceScores, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detectedClassNames.append(name)

            # Показ імені класу та точності на відео
            cv2.putText(img, f'{name.upper()} {int(confidenceScores[i] * 100)}%',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Малювання обмежувального квадрата
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, requiredClassIndex.index(classIds[i])])

    # Оновлення трекеру для кожного об'єкта
    boxesIds = tracker.update(detection)
    for boxId in boxesIds:
        countVehicle(boxId, img)


def realTime():
    i = 0
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter.fourcc(*'DIVX'), 15.0, (320,200))

    while True and i < frames:
        success, img = cap.read()
        img = cv2.resize(img, (0, 0), None, 1, 1)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (inputSize, inputSize), [0, 0, 0], 1, crop=False)

        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]

        # Постачання даних до мережі0
        outputs = net.forward(outputNames)

        # Пошук об'єктів із мережевого виводу
        postProcess(outputs, img)

        # Малювання ліній перетину
        cv2.line(img, (0, middleLinePosition), (iw, middleLinePosition), (255, 0, 255), 2)
        cv2.line(img, (0, upLinePosition), (iw, upLinePosition), (0, 0, 255), 2)
        cv2.line(img, (0, downLinePosition), (iw, downLinePosition), (0, 0, 255), 2)

        # Показ поточних результатів на відео
        # cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Car:        " + str(up_list[0]) + "     " + str(down_list[0]), (20, 40),
        #            cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Motorbike:  " + str(up_list[1]) + "     " + str(down_list[1]), (20, 60),
        #            cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Bus:        " + str(up_list[2]) + "     " + str(down_list[2]), (20, 80),
        #            cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # cv2.putText(img, "Truck:      " + str(up_list[3]) + "     " + str(down_list[3]), (20, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Вивід відео
        out.write(img)
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

        i += 1

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        upList.insert(0, "Up")
        downList.insert(0, "Down")
        cwriter.writerow(upList)
        cwriter.writerow(downList)
    f1.close()

    print("Результати успішно збережені у файл 'data.csv'")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realTime()
