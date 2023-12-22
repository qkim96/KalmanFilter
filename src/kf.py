import numpy as np
from numpy.linalg import inv
import imutils
from imutils.video import VideoStream
from pyautogui import size as sz
import cv2
import time


def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# set variables for Kalman filter
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

# set variables for video/image processing
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
found = False
vid = VideoStream(src=0, framerate=10).start()

# wait for the camera
time.sleep(2.0)

while True:
    start = time.time()
    frame = vid.read()

    if frame is None:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame = imutils.resize(frame, width=int(sz()[0]*0.9))
    blr = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cntr = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntr = imutils.grab_contours(cntr)

    # if detected, draw red circle (measured)
    if len(cntr) > 0:
        found = True
        c = max(cntr, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # if detected at least once, draw blue circle (predicted)
    # with measurement (visible)
    if found and (len(cntr) > 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([[x, y]]))
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (255, 0, 0), -1)
    # without measurement (occluded)
    elif found and (len(cntr) <= 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]
        cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (255, 0, 0), 2)
        cv2.circle(frame, (int(x_bel), int(y_bel)), 5, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)
    time.sleep(max(1. / 25 - (time.time() - start), 0))  # match fps of camera and while loop

vid.stop()
cv2.destroyAllWindows()
