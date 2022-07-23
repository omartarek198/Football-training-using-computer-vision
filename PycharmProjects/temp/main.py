import cv2
from LiveTimeExtraction import *
import VideoExtraction

obj = LTextraction(2)

# obj.LiveStream(500)


objv = VideoExtraction.VideoExtractor()
landmarks,frames = objv.GetLandMarksInVideo('Videos/jump1.mp4')
print(landmarks)
print(frames)



