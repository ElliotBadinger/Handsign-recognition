import cv2 #Display image in a window
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier 
import numpy as np # perform a wide variety of math operations on arrays 
import math

# Constants
OFFSET = 20
IMG_SIZE = 300
FOLDER = "Data/C"
LABELS = ["A", "B", "C"]
MAX_HANDS = 2
MODEL_PATH = "Model/keras_model.h5"
LABELS_PATH = "Model/labels.txt"

def setup():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=MAX_HANDS)
    classifier = Classifier(MODEL_PATH, LABELS_PATH)
    return cap, detector, classifier

def process_frame(detector, classifier, img):
    img_output = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        x_start = max(0, x - OFFSET)
        y_start = max(0, y - OFFSET)
        x_end = min(img.shape[1], x + w + OFFSET)
        y_end = min(img.shape[0], y + h + OFFSET)
        img_crop = img[y_start:y_end, x_start:x_end]
        aspect_ratio = h / w
        prediction, index = resize_and_predict(img_crop, aspect_ratio, classifier)
        draw_on_image(img_output, x, y, w, h, index)
    return img_output

def resize_and_predict(img_crop, aspect_ratio, classifier):
    if aspect_ratio > 1:
        scaling_factor = IMG_SIZE / img_crop.shape[0]
        calculated_width = math.ceil(scaling_factor * img_crop.shape[1])
        img_resize = cv2.resize(img_crop, (calculated_width, IMG_SIZE))
        w_gap = math.ceil((IMG_SIZE - calculated_width) / 2)
        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        img_white[:, w_gap:calculated_width + w_gap] = img_resize
    else:
        scaling_factor = IMG_SIZE / img_crop.shape[1]
        calculated_height = math.ceil(scaling_factor * img_crop.shape[0])
        img_resize = cv2.resize(img_crop, (IMG_SIZE, calculated_height))
        h_gap = math.ceil((IMG_SIZE - calculated_height) / 2)
        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        img_white[h_gap:calculated_height + h_gap, :] = img_resize
    prediction, index = classifier.getPrediction(img_white, draw=False)
    return prediction, index

def draw_on_image(img, x, y, w, h, index):
    cv2.rectangle(img, (x - OFFSET, y - OFFSET-50), (x - OFFSET+90, y - OFFSET-50+50), (255, 0, 255), cv2.FILLED)
    cv2.putText(img, LABELS[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
    cv2.rectangle(img, (x-OFFSET, y-OFFSET), (x + w+OFFSET, y + h+OFFSET), (255, 0, 255), 4)

def main():
    cap, detector, classifier = setup()
    while True:
        success, img = cap.read()
        if not success:
            break
        img_output = process_frame(detector, classifier, img)
        cv2.imshow("Image", img_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
