import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from keras.models import load_model

# loading pre trained model
model = load_model('cnn_model/digit_classifier.h5')

# Reshaping to feed our image to the model and get prediction
def predict_digit(img):
    test_image = img.reshape(-1, 28, 28, 1)
    return np.argmax(model.predict(test_image))


# putting label on each image
# https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
# https: // www.geeksforgeeks.org / python - opencv - cv2 - rectangle - method /
def put_label(t_img, label, x, y):
    font = cv2.FONT_HERSHEY_TRIPLEX
    l_x = int(x) - 12
    l_y = int(y) - 10
    cv2.rectangle(t_img, (l_x, l_y + 5), (l_x + 35, l_y - 35), (128, 204, 255), -1)
    cv2.putText(t_img, str(label), (l_x, l_y), font, 1.5, (255, 0, 0), 1, cv2.LINE_AA)
    return t_img

# https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
# refining each digit by adding padding
def image_refiner(gray):
    org_size = 22
    img_size = 28
    rows, cols = gray.shape
    print(" row:{}, col:{} ".format(rows, cols))
    # Maintaining the Aspect Ratio
    if rows > cols:
        factor = org_size / rows
        rows = org_size
        cols = int(round(cols * factor))
    else:
        factor = org_size / cols
        cols = org_size
        rows = int(round(rows * factor))

    print(" row:{}, col:{} ".format(rows, cols))
    gray = cv2.resize(gray, (cols, rows))
    print(" row:{}, col:{} ".format(rows, cols))
    print(gray.shape)
    # get padding
    colsPadding = (int(math.ceil((img_size - cols) / 2.0)), int(math.floor((img_size - cols) / 2.0)))
    rowsPadding = (int(math.ceil((img_size - rows) / 2.0)), int(math.floor((img_size - rows) / 2.0)))

    # apply padding
    # https://www.geeksforgeeks.org/numpy-pad-function-in-python/#:~:text=pad()%20function%20is%20used,will%20increase%20according%20to%20pad_width.
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')
    print("NEW: ", gray.shape)
    return gray


    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html
    # https: // www.geeksforgeeks.org / enumerate - in -python /
def get_output_image(path):
    img = cv2.imread(path, 0)
    img_org = cv2.imread(path)

    #cv2.imshow('dig_imagage',img)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html

    for j, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        #hieararcy !=-1 because we are not interested in outeremost contours i.e. the border
        if (hierarchy[0][j][3] != -1 and w > 10 and h > 10):
            # putting boundary on each digit
            cv2.rectangle(img_org, (x, y), (x + w, y + h), (0, 64, 225), 2)

            # cropping each image and process
            roi = img[y:y + h, x:x + w]
            # bitwise_not changes the white into black and black into white
            roi = cv2.bitwise_not(roi)
            roi = image_refiner(roi)

            # getting prediction of cropped image
            pred = predict_digit(roi)
            print(pred)

            # placing label on each digit
            # (x, y), radius = cv2.minEnclosingCircle(cnt)
            x = x+w/2
            img_org = put_label(img_org, pred, x, y)

    return img_org