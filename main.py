import numpy as np
from cv2 import cv2


class Side(object):
    def __init__(self, pathImg, pathCalibration):
        self.capture = cv2.VideoCapture(pathImg)
        s = cv2.FileStorage()
        s.open(pathCalibration, cv2.FILE_STORAGE_READ)
        self.scaled = s.getNode("Scale")
        self.camMat = s.getNode("CameraMatrix").mat()
        self.distCoef = s.getNode("DistortionCoeffs").mat()
        self.w = int(s.getNode("Width").real())
        self.h = int(s.getNode("Height").real())

    def getImage(self):
        rat, img = self.capture.read()
        assert rat
        cv2.resize(img, (self.w // 4, self.h // 4), cv2.INTER_AREA)
        self.w /= 4
        self.w /= 4
        return img

    def getUndistortImage(self):
        rat, img = self.capture.read()
        assert rat
        nk = self.camMat.copy()
        nk[0, 0] /= 1
        nk[1, 1] /= 1
        cv2.resize(img, (self.w, self.h), cv2.INTER_AREA)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.camMat, self.distCoef, np.eye(3), nk, (self.w, self.h),
                                                         cv2.CV_16SC2)
        img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return img


class Zone(object):

    def __init__(self, topLeft, bottomLeft, bottomRight, topRight, color):
        self.topLeft = topLeft
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
        self.topRight = topRight
        self.color = color

    def getRect(self):
        return np.float32([self.topLeft, self.bottomLeft, self.bottomRight, self.topRight])

    def paintLines(self, image):
        cv2.line(image, self.topLeft, self.bottomLeft, self.color, 1)
        cv2.line(image, self.bottomLeft, self.bottomRight, self.color, 1)
        cv2.line(image, self.bottomRight, self.topRight, self.color, 1)
        cv2.line(image, self.topRight, self.topLeft, self.color, 1)

    def warpPerspective(self, image):
        M = cv2.getPerspectiveTransform(self.getRect(),
                                        np.float32([(0, 0), (0, 540), (240, 540), (240, 0)]))
        return cv2.warpPerspective(image, M, (240, 540), 1)


if __name__ == '__main__':
    sideRight = Side("resources/video/record-2018-07-25-13-51-49_0.mkv", "resources/calibration/11-09-0a-0f-05-09.yaml")
    sideLeft = Side("resources/video/record-2018-07-25-13-51-49_1.mkv", "resources/calibration/04-1e-09-08-02-07.yaml")
    sideFront = Side("resources/video/record-2018-07-25-13-51-49_2.mkv", "resources/calibration/11-14-0e-0f-05-09.yaml")
    sideRear = Side("resources/video/record-2018-07-25-13-51-49_3.mkv", "resources/calibration/04-1e-1b-08-02-07.yaml")
    i = 1
    out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))
    while i <= 10000:
        imgRight = sideRight.getUndistortImage()
        imgLeft = sideLeft.getUndistortImage()
        imgFront = sideFront.getUndistortImage()
        imgRear = sideRear.getUndistortImage()



        # dst = np.concatenate((img, dst1, dst2, dst3, dst4), 1)
        cv2.imshow("image", imgRight)
        cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
