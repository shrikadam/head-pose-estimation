from head_pose_module import *

cap = cv2.VideoCapture(2)  # From Camera

while(cap.isOpened()):

    # Take each frame
    ret, img = cap.read()
    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        img_h, img_w, img_c = img.shape
        text = ''
        res = process_frame(img)
        cv2.imshow('Head Pose', res)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()