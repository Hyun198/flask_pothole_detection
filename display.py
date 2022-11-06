import cv2


cap = cv2.VideoCapture("video.mp4")

while(cap.isOpened()):
    ret,frame = cap.read()

    frame = cv2.resize(frame,(1200,700))

    cv2.imshow("video",frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()