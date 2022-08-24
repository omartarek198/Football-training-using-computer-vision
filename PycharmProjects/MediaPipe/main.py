import cv2
import mediapipe as mp
from dollarpy import Recognizer, Template, Point
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from tslearn.metrics import dtw, dtw_path
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from tslearn import metrics


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('videos/sitting1.mp4')
lmlist = []
lmlistl = []
klist =[]
xlist = []
ylist = []
label = []
while True:
    success, img = cap.read()
    success, frames = cap.read()
    try:
        imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            if id == 25 or id == 26:
                lmlist.append(Point(lm.x,lm.y))
                lmlistl.append('sitting')
                label.append('sitting')
                xlist.append(lm.x)
                ylist.append(lm.y)
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
print('DataFrame is written to Excel File successfully.')


tmpl_1 = Template('sitting', lmlist)

cap = cv2.VideoCapture('videos/longwalk.mp4')
lmlist2 = []
lmlist2l = []

while True:
    success, img = cap.read()
    success, frames = cap.read()
    try:
        imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            if id == 25 or id == 26:
                # print(id,lm)
                lmlist2.append(Point(lm.x,lm.y))
                lmlist2l.append('walking')
                label.append('walking')
                xlist.append(lm.x)
                ylist.append(lm.y)

    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

tmpl_2 = Template('walking', lmlist2)


cap = cv2.VideoCapture('videos/shortwalk.mp4')
lmlist3 = []
lmlist3l = []
while True:
    success, img = cap.read()
    success, frames = cap.read()
    try:
        imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            if id == 25 or id == 26:
                # print(id,lm)
                lmlist3.append(Point(lm.x,lm.y))
                lmlist3l.append('walking')
                label.append('walking')
                xlist.append(lm.x)
                ylist.append(lm.y)
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

tmpl_3 = Template('walking', lmlist3)


cap = cv2.VideoCapture('videos/sitting2.mp4')
lmlist4 = []
lmlist4l = []
while True:
    success, img = cap.read()
    success, frames = cap.read()
    try:
        imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            if id == 25 or id == 26:
                # print(id,lm)
                lmlist4.append(Point(lm.x,lm.y))
                lmlist4l.append('sitting')
                label.append('sitting')
                xlist.append(lm.x)
                ylist.append(lm.y)
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

tmpl_4 = Template('sitting', lmlist4)

cap = cv2.VideoCapture('videos/sitting3.mp4')
lmlist5 = []
lmlist5l =[]
temp1 = []
while True:
    success, img = cap.read()
    success, frames = cap.read()
    try:
        imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            if id == 25 or id == 26:
                # print(id,lm)
                lmlist5.append(Point(lm.x,lm.y))
                lmlist5l.append('sitting')
                label.append('sitting')
                xlist.append(lm.x)
                ylist.append(lm.y)
                temp1.append(lm.x)
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

tmpl_5 = Template('sitting', lmlist5)
temp2=[]
cap = cv2.VideoCapture('videos/Walk.mp4')
lmlist6 = []
lmlist6l = []
xlist2 = []
ylist2 = []
while True:
    success, img = cap.read()
    success, frames = cap.read()
    try:
        imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    except:
        break
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            if id == 25 or id == 26:
                # print(id,lm)
                lmlist6.append(Point(lm.x,lm.y))
                lmlist6l.append('walking')
                xlist2.append(lm.x)
                ylist2.append(lm.y)
                temp2.append(lm.x)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()



recognizer = Recognizer([tmpl_1,tmpl_2,tmpl_3,tmpl_4,tmpl_5])

result = recognizer.recognize(lmlist6)
print(result)
# df = pd.DataFrame({'x': xlist2,
#                    'y': ylist2,
#                    })
#
# # creating the DataFrame
# file_name = 'Test.xslx'
#
# # saving the excel
# df.to_excel(file_name)
# print('DataFrame is written to Excel File successfully.')

columns = ['x','y','label']
dftrain = pd.read_csv('Train.csv', usecols=columns)


X_train, X_test, y_train, y_test = train_test_split(dftrain.drop('label',axis='columns'), dftrain.label, test_size=0.3,random_state=10)

#feature based ext.
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(knn.score(X_test,y_test))

#dtw

x = X_train
y = X_test
dtw_score = dtw(x, y)
# Or, if the path is also an important information:
optimal_path, dtw_score = dtw_path(x, y)

from tslearn.metrics import dtw_path_from_metric


from tslearn.metrics import dtw
cost = dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)

print(dtw_score)
print(cost)
from tslearn.metrics import dtw
cost = dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
print(cost)
from tslearn.metrics import dtw
cost = dtw(x, y, global_constraint="itakura", itakura_max_slope=2.)
print(cost)
from tslearn.metrics import soft_dtw
soft_dtw_score = soft_dtw(x, y, gamma=.1)
print(soft_dtw_score)