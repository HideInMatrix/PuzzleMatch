import cv2
import numpy as np


class PuzzleMatch:
    def __init__(self):
        super(self)

    @staticmethod
    def watershed_fn(filepath):
        # 读取灰度图像
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # 找到最亮的点
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)

        # 定义一个亮度范围，例如，以最亮点的亮度为中心，定义一个范围
        brightness_range = 10
        lower_bound = max_val - brightness_range
        upper_bound = max_val + brightness_range

        # 创建一个二进制图像，其中位于亮度范围内的像素值为255，其他为0
        mask = cv2.inRange(img, lower_bound, upper_bound)

        # 找到符合条件的像素的坐标
        bright_pixels = np.column_stack(np.where(mask > 0))

        # 在原始图像上标记符合条件的点为绿色
        img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for pixel in bright_pixels:
            cv2.circle(img_colored, tuple(pixel[::-1]), 1, (0, 255, 0), -1)

        # 在最亮的点处画一个红色的圆
        cv2.circle(img_colored, max_loc, 1, (0, 0, 255), -1)

        # 创建一个黑色图像，与原始图像大小相同
        black_img = np.zeros_like(img_colored)

        # 将红色和绿色的像素复制到黑色图像
        black_img[(img_colored == [0, 255, 0]).all(axis=-1) | (img_colored == [0, 0, 255]).all(axis=-1)] = img_colored[
            (img_colored == [0, 255, 0]).all(axis=-1) | (img_colored == [0, 0, 255]).all(axis=-1)]

        return black_img


def find_most_probable_shape(black_img):
    # 将黑白图像转换为灰度图像
    gray_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)

    # 使用阈值处理将图像转换为二进制图像
    _, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最大轮廓和最大轮廓面积
    max_contour = None
    max_area = 0

    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 更新最大轮廓和最大面积
        if area > max_area:
            max_area = area
            max_contour = contour

    # 在原始图像上绘制最大轮廓
    result_img = cv2.drawContours(black_img.copy(), [max_contour], -1, (255, 0, 0), 2)

    return result_img


if __name__ == "__main__":
    img = PuzzleMatch.watershed_fn('back.jpeg')
    rest_img = find_most_probable_shape(img)
    # 显示结果
    cv2.imshow("Result", rest_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
