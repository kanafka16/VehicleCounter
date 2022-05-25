import math


class EuclideanDistTracker:
    def __init__(self):
        # Зберігання центральних позицій об'єктів
        self.centerPoints = {}
        # Зберігання кількості ID
        # При кожному виявленні об'єкту з новим ID кількість буде зростати
        self.idCount = 0

    def update(self, objectsRect):
        # ID та квадрати об'єктів
        objectsBbsIds = []

        # Отримання точки центру об'єктів
        for rect in objectsRect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Визначення чи об'єкт вже знайдений
            sameObjectDetected = False
            for id, pt in self.centerPoints.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.centerPoints[id] = (cx, cy)

                    objectsBbsIds.append([x, y, w, h, id, index])
                    sameObjectDetected = True

                    break

            # Присвоєння ID новому об'єкту
            if sameObjectDetected is False:
                self.centerPoints[self.idCount] = (cx, cy)
                objectsBbsIds.append([x, y, w, h, self.idCount, index])
                self.idCount += 1

        # Очищення словника від непотрібних ID точками центрів
        newCenterPoints = {}
        for objBbId in objectsBbsIds:
            _, _, _, _, objectId, index = objBbId
            center = self.centerPoints[objectId]
            newCenterPoints[objectId] = center

        # Оновлення словника з не використаними або видаленими ID
        self.centerPoints = newCenterPoints.copy()

        return objectsBbsIds


def ad(a, b):
    return a + b
