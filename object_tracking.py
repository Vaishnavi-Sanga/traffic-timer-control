import cv2
import numpy as np
from object_detection import ObjectDetection

od = ObjectDetection()

cap = cv2.VideoCapture("video5.mp4")

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)

#fc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.mp4', fc, 5, (500, 500))

s = 20
cl = 0
cr = 0
l1 = []
l2 = []
while (s):
    ret, frame = cap.read()

    height, width, _ = frame.shape
    #print(height, width)

    r1 = frame[:, : width//2]
    r2 = frame[:, width//2:]

    if not ret:
        break

    #b = cv2.resize(frame, (500,500), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
    #out.write(b)

    (class_ids, scores, boxes) = od.detect(r1)
    c1 = 0
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(r1, (x,y), (x+w, y+h), (255,255,0), 2)
        c1 = c1 + 1
    print("Vehicle count on left: ", c1)
    l1.append(c1)

    (class_ids, scores, boxes) = od.detect(r2)
    c2 = 0
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(r2, (x,y), (x+w, y+h), (255,255,0), 2)
        c2 = c2 + 1
    print("Vehicle count on right: ", c2)
    l2.append(c2)

    cv2.imshow("LEFT ", r1)
    cv2.imshow("RIGHT ", r2)
    #cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if(c1 > c2):
        cl = cl + 1
    else:
        cr = cr + 1
    s = s - 1

tl, tr = 1, 1

if(cl > cr):
    print("Left side first")
    t = max(l1)//max(l2)
    tl = t*tr
else:
    print("Right side first")
    t = max(l2)//max(l1)
    tr = t*tl

print("Time on left side: ", tl)
print("Time on right side: ", tr)


cap.release()
cv2.destroyAllWindows()
