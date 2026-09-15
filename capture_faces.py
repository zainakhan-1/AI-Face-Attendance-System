import cv2
import os
import time

def capture_faces(name):
    save_path = f"dataset/{name}"
    os.makedirs(save_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    for i in range(5, 0, -1):
        ret, frame = cam.read()
        if not ret:
            continue
        cv2.putText(frame, f"Starting in {i}...", (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Get Ready", frame)
        cv2.waitKey(1000)

    cv2.destroyWindow("Get Ready")

    count = 0
    while count < 20:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.putText(frame, f"Capturing {count+1}/20", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capturing", frame)
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        count += 1
        time.sleep(0.5)
        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
