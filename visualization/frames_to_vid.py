import cv2
import os,re


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


root = 'runs/synthetic/exp8'
images = [img for img in humanSort(os.listdir(root)) if img.endswith(".png")]
frame = cv2.imread(os.path.join(root, images[0]))
height, width, layers = frame.shape
print(height, width)
video = cv2.VideoWriter("videos/3-synthetic-check-time.mp4", cv2.VideoWriter_fourcc(*'mp4v'),  30 , (width,height))

full = len(images)
for i, image in enumerate(images):
    video.write(cv2.imread(os.path.join(root, image)))
    print(i,full)