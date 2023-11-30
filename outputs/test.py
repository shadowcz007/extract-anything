import cv2
import numpy as np
from PIL import Image,ImageFilter



def fill(img):
    """
    形态学重建之孔洞填充。
    :param img_path: 输入图像的路径
    :param out_path: 需要保存图像的路径根路径。
    :return:
    """
    # 读取并处理图像
    img = img.convert('L')
    img = np.array(img)

    # 对原图取反，‘255’可以根据图像中类别值进行修改。（例如，图像中二值为0和1，那么255则修改为1）
    # 此时mask用来约束膨胀结果。原图白色为边界，黑色为孔洞和背景，取反后黑色为边界，白色为孔洞和背景。
    mask = 255 - img
 
    # 以带有白色边框的黑色图像为初始Marker，用来SE来连续膨胀，该图通过迭代生成填充图像。
    marker = np.zeros_like(img)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255

    # 形态学重建
    SE = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
    count = 0
    while True:
        count += 1
        marker_pre = marker
        # 膨胀marker
        dilation = cv2.dilate(marker, kernel=SE)
        # 和mask进行比对，用来约束膨胀。由于mask中黑色为边界，白色为孔洞和背景。
        # 当遇到黑色边界后，就无法继续继续前进膨胀。当遇到白色后，会继续向里面膨胀。孔洞闭合的，遇到全部的黑色边界后，内部自然就已经被填充。
        marker = np.min((dilation, mask), axis=0)
      
        # 判断经过膨胀后的结果是否和上次迭代一致，如果一致则完成孔洞填充。
        if (marker_pre == marker).all():
            break

    # 将结果取反，还原为原来的图像情况。即白色为边界，黑色为孔洞和背景，
    dst = 255 - marker

    # 保存结果图像
    # Image.fromarray(dst).save(out_path)
    return Image.fromarray(dst)


# res=fill(Image.open('mask.png'))
# res.save('out_path.png')
def smooth_filter(image, radius):
    # Apply smoothing filter to the image
    smoothed_image = cv2.blur(image, (radius, radius))
    return smoothed_image

# Example usage
image = cv2.imread('mask.png')  # Replace 'image.jpg' with your image file path
radius = 5  # Replace with your desired radius

smoothed_image = smooth_filter(image, radius)
cv2.imshow('Smoothed Image', smoothed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


