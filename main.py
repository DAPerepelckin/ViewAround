import numpy as np
from cv2 import cv2
from imutils import paths
import imutils


class Side(object):
    def __init__(self, pathImg, pathCalibration):
        self.capture = cv2.VideoCapture(pathImg)
        s = cv2.FileStorage()
        s.open(pathCalibration, cv2.FILE_STORAGE_READ)
        self.camMat = s.getNode("CameraMatrix").mat()
        self.distCoef = s.getNode("DistortionCoeffs").mat()
        self.w = int(s.getNode("Width").real())
        self.h = int(s.getNode("Height").real())

    def getImage(self):
        rat, img = self.capture.read()
        if rat:
            cv2.resize(img, (self.w // 4, self.h // 4), cv2.INTER_AREA)
            self.w /= 4
            self.w /= 4
        return rat, img

    def getUndistortImage(self):
        rat, img = self.capture.read()
        if rat:
            nk = self.camMat.copy()
            nk[0, 0] /= 1
            nk[1, 1] /= 1
            cv2.resize(img, (self.w, self.h), cv2.INTER_AREA)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.camMat, self.distCoef, np.eye(3), nk,
                                                             (self.w, self.h),
                                                             cv2.CV_16SC2)
            img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return rat, img


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
                                        np.float32([(600, 0), (0, 1080), (1920, 1080), (1320, 0)]))
        return cv2.warpPerspective(image, M, (1920, 1080), 1)

#(0, 0), (0, 880), (1920, 880), (1920, 0)
# (624, 434), (4, 648), (1430, 708), (1091, 436)
def viewVideo():
    sideRight = Side("resources/video/record-2018-07-25-13-51-49_0_Trim.mp4",
                     "resources/calibration/11-09-0a-0f-05-09.yaml")
    # sideLeft = Side("resources/video/record-2018-07-25-13-51-49_1.mkv", "resources/calibration/04-1e-09-08-02-07.yaml")
    # sideFront = Side("resources/video/record-2018-07-25-13-51-49_2.mkv", "resources/calibration/11-14-0e-0f-05-09.yaml")
    # sideRear = Side("resources/video/record-2018-07-25-13-51-49_3.mkv", "resources/calibration/04-1e-1b-08-02-07.yaml")

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (1920, 1080))

    while sideRight.capture.isOpened():
        rat, imgRight = sideRight.getUndistortImage()
        if rat:
            # imgLeft = sideLeft.getUndistortImage()
            # imgFront = sideFront.getUndistortImage()
            # imgRear = sideRear.getUndistortImage()
            # dst = np.concatenate((img, dst1, dst2, dst3, dst4), 1)

            out.write(imgRight)

            # cv2.imshow("image", imgRight)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
    sideRight.capture.release()
    out.release()
    cv2.destroyAllWindows()


def right():

    s = cv2.FileStorage()
    s.open("resources/calibration/11-09-0a-0f-05-09.yaml", cv2.FILE_STORAGE_READ)
    camMat = s.getNode("CameraMatrix").mat()
    distCoef = s.getNode("DistortionCoeffs").mat()
    w = int(s.getNode("Width").real())
    h = int(s.getNode("Height").real())
    img = cv2.imread("resources/video/record-2018-07-25-13-51-49_0_Moment.jpg")

    nk = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camMat, distCoef, (1920, 1080), np.eye(3), balance=0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(camMat, distCoef, np.eye(3), nk,
                                                     (1920, 1080),
                                                     cv2.CV_16SC2)
    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    scale_percent = 100  # Процент от изначального размера
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LANCZOS4)
    return resized


def rear():

    s = cv2.FileStorage()
    s.open("resources/calibration/04-1e-1b-08-02-07.yaml", cv2.FILE_STORAGE_READ)
    camMat = s.getNode("CameraMatrix").mat()
    distCoef = s.getNode("DistortionCoeffs").mat()
    w = int(s.getNode("Width").real())
    h = int(s.getNode("Height").real())
    img = cv2.imread("resources/video/record-2018-07-25-13-51-49_3_Moment.jpg")
    nk = camMat.copy()
    nk[0, 0] /= 1
    nk[1, 1] /= 2
    cv2.resize(img, (w, h), cv2.INTER_AREA)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(camMat, distCoef, np.eye(3), nk,
                                                     (w, h),
                                                     cv2.CV_16SC2)
    img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    scale_percent = 50  # Процент от изначального размера
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def viewImage():
    img_ = right()
    # img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = rear()

    # img = cv2.resize(img, (0,0), fx=1, fy=1)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)



    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)
    img3 = cv2.drawMatches(img_, kp1, img, kp2, good, None, **draw_params)
    cv2.imshow("original_image_drawMatches.jpg", img3)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow("original_image_overlapping.jpg", img2)

    # dst = np.concatenate((rightImg, rearImg), 1)
    # cv2.imshow("Image", )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_ = right()
    cv2.imshow("original_image_overlapping.jpg", img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #viewImage()
