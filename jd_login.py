import base64

import cv2
import numpy as np


def get_object_bounding_box(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行阈值处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 对阈值图像进行开运算
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # 进行背景的膨胀操作
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # 寻找前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # 寻找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # 标记未知区域
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # 进行分水岭算法
    markers = cv2.watershed(image, markers)
    # 绘制边框
    contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for i in range(len(contours)):
        # 忽略外部轮廓
        if hierarchy[0][i][3] == -1:
            x, y, w, h = cv2.boundingRect(contours[i])
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes


def match_puzzle(background_img, miss_puzzle):
    # 加载源图片和模板图片
    img_rgb = cv2.imread(background_img)
    template = cv2.imread(miss_puzzle)

    # 读取模板图片的宽度和高度
    w, h = template.shape[:-1]

    # 匹配模板
    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)

    # 设定阈值
    threshold = 0.8

    # 找到匹配区域
    loc = np.where(res >= threshold)

    # 在源图片上用矩形框标出匹配区域
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    # 显示图片
    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    match_puzzle('back.jpeg', 'puzzle,png')