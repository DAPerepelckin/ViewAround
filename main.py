import numpy as np
from cv2 import cv2

if __name__ == '__main__':
    capture = cv2.VideoCapture("resources/video/record-2018-07-25-13-51-49_0.mkv")
    # img = cv2.imread("resources/screenshot.png")
    s = cv2.FileStorage()
    s.open("resources/calibration/11-09-0a-0f-05-09.yaml", cv2.FILE_STORAGE_READ)
    scaled = s.getNode("Scale")
    camMat = s.getNode("CameraMatrix").mat()
    distCoef = s.getNode("DistortionCoeffs").mat()
    w = 1920
    h = 1080
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camMat, distCoef, (w, h), 1, (960, 540))
    # new h = 540 w = 960
    while True:
        rat, img = capture.read()

        img = cv2.undistort(img, camMat, distCoef, None, newCameraMatrix)
        img = img[0:540, 0:960]

        cv2.line(img, (300, 180), (140, 390), (10, 155, 255), 1)
        cv2.line(img, (140, 390), (260, 475), (10, 155, 255), 1)
        cv2.line(img, (260, 475), (390, 170), (10, 155, 255), 1)

        cv2.line(img, (390, 170), (260, 475), (0, 255, 255), 1)
        cv2.line(img, (260, 475), (560, 520), (0, 255, 255), 1)
        cv2.line(img, (560, 520), (510, 170), (0, 255, 255), 1)

        cv2.line(img, (510, 170), (560, 520), (55, 255, 110), 1)
        cv2.line(img, (560, 520), (790, 445), (55, 255, 110), 1)
        cv2.line(img, (790, 445), (615, 175), (55, 255, 110), 1)

        cv2.line(img, (615, 175), (790, 445), (255, 255, 0), 1)
        cv2.line(img, (790, 445), (890, 360), (255, 255, 0), 1)
        cv2.line(img, (890, 360), (710, 190), (255, 255, 0), 1)

        M1 = cv2.getPerspectiveTransform(np.float32([(300, 180), (140, 390), (260, 475), (390, 170)]),
                                         np.float32([(0, 0), (0, 540), (270, 540), (270, 0)]))

        M2 = cv2.getPerspectiveTransform(np.float32([(390, 170), (260, 475), (560, 520), (510, 170)]),
                                         np.float32([(0, 0), (0, 540), (270, 540), (270, 0)]))

        M3 = cv2.getPerspectiveTransform(np.float32([(510, 170), (560, 520), (790, 445), (615, 175)]),
                                         np.float32([(0, 0), (0, 540), (270, 540), (270, 0)]))

        M4 = cv2.getPerspectiveTransform(np.float32([(615, 175), (790, 445), (890, 360), (710, 190)]),
                                         np.float32([(0, 0), (0, 540), (270, 540), (270, 0)]))

        dst1 = cv2.warpPerspective(img, M1, (270, 540), 1)
        dst2 = cv2.warpPerspective(img, M2, (270, 540), 1)
        dst3 = cv2.warpPerspective(img, M3, (270, 540), 1)
        dst4 = cv2.warpPerspective(img, M4, (270, 540), 1)

        dst = np.concatenate((img, dst1, dst2, dst3, dst4), 1)
        cv2.imshow("image1", dst)
        cv2.waitKey(10)
