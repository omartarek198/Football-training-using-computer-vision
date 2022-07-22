# The current data set is huge as a result i will filter it using this script


import os
import shutil

data_path = "D:\CV projects\data"
destination = "D:/CV projects/Football-training-using-computer-vision/filtered data"
files = os.listdir(data_path)

tot = 0
temp = "\\"
for file in files:
    name = file.split("_")
    if name[2] == "a01" or name[2] == "a04":
        shutil.move(data_path +temp+ file, destination+temp+file)
        print(file)
        tot += 1

print(tot)
