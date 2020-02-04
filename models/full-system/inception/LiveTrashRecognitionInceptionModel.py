import cv2

# the webcam
cv2.namedWindow("Live Screen")
cv2.namedWindow("Resized")
vc = cv2.VideoCapture(0)

while True:
    # capture the frame
    ret, frame = vc.read()

    # resize the image into 244 x 244
    resized_frame = cv2.resize(frame, (244, 244))
    cv2.imshow("Resized", resized_frame)

    cv2.imshow("Live Screen", frame)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

# exit code
vc.release()
cv2.destroyAllWindows()