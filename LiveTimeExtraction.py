import cv2
from Extractor import *

class LTextraction:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.Extractor = CExtractor()
    def LiveStream(self, duration):
        frames = []
        counter = 0
        cap = cv2.VideoCapture(self.cam_id)
        while counter < duration:

            sucess, image = cap.read()

            cv2.imshow("LiveTracking" , self.Extractor.detectPose(image))
            frames.append(image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            counter +=1


        cap.release()
        print(len(frames))
        return frames
