from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
help="path to input image")
args = vars(ap.parse_args())



img_array = cv2.imread(args["image"])
model = load_model(args["model"])

orig = img_array.copy()# convert to array
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

img_rgb = cv2.resize(img_rgb,(224,224),3)  # resize
img_rgb = np.array(img_rgb).astype(np.float32)/255.0  # scaling
img_rgb = np.expand_dims(img_rgb, axis=0)  # expand dimension
y_pred = model.predict(img_rgb) # prediction
y_pred_class = y_pred.argmax(axis=1)[0]
import numpy as np
import pandas as pd
prediction = pd.DataFrame(y_pred_class, columns=['predictions']).to_csv('prediction.csv')

if y_pred_class==0:
    label="c0: safe driving"

if y_pred_class==1:
    label="c1: texting — right" 

if y_pred_class==2:
    label=" c2: talking on the phone — right"

if y_pred_class==3:
    label="c3: texting — left"
    
if y_pred_class==4:
    label="c4: talking on the phone — left"

if y_pred_class==5:
    label="c5: operating the radio"

if y_pred_class==6:
    label="c6: drinking"

if y_pred_class==7:
    label="c7: reaching behind"

if y_pred_class==8:
    label="c8: hair and makeup"

if y_pred_class==9:
    label=" c9: talking to a passenger"
    
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
0.7, (0, 255, 0), 2)
cv2.imshow("Output", output)

cv2.waitKey(0)





