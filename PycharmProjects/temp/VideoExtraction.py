import cv2
import mediapipe as mp
from Extractor import *
class VideoExtractor:
    def __init__(self):
        self.Extractor = CExtractor()
    def GetLandMarksInVideo(self,videoPath):
        mpPose = mp.solutions.pose
        mpDraw = mp.solutions.drawing_utils
        pose = mpPose.Pose()
        cap = cv2.VideoCapture(videoPath)
        lmlist = []
        frameslist = []
        while cap.isOpened():
            success, frames = cap.read()
            try:
                imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
            except:
                break
            frameslist.append(frames)
            results = pose.process(imgRGB)
            mpDraw.draw_landmarks(frames, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for lm in enumerate(results.pose_landmarks.landmark):
                x, y , z = frames.shape
                lmlist.append([x , y , z])
            cv2.imshow("Video", frames)
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
        return lmlist , frameslist

