import cv2
import numpy as np
from PIL import Image,ImageOps

def process_image(file_path, start_offset=32, feathering_weight=0.8):
    # Open the image using PIL
    image = Image.open(file_path).convert("L")
    if start_offset<0:
        image=ImageOps.invert(image)

    # Convert the image to a numpy array
    image_np = np.array(image)

    # Use Canny edge detection to get black contours
    edges = cv2.Canny(image_np, 30, 150)

    for i in range(0,abs(start_offset)):
        # int(100*feathering_weight)
        a=int(abs(start_offset)*0.1*i)
        # Dilate the black contours to make them wider
        kernel = np.ones((a, a), np.uint8)

        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # dilated_edges = cv2.erode(edges, kernel, iterations=1)
        # Smooth the dilated edges using Gaussian blur
        smoothed_edges = cv2.GaussianBlur(dilated_edges, (5, 5), 0)

        # Adjust the feathering weight
        feathering_weight = max(0, min(feathering_weight, 1))

        # Blend the smoothed edges with the original image to achieve feathering effect
        image_np = cv2.addWeighted(image_np, 1, smoothed_edges, feathering_weight, feathering_weight)

    # Convert the result back to PIL image
    result_image = Image.fromarray(np.uint8(image_np))
    if start_offset<0:
        result_image=ImageOps.invert(result_image)
    
    return result_image



def crop_image_remove_transparent(image):
    # 将PIL的Image类型转换为OpenCV的numpy数组
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

    # 分离图像的RGBA通道
    rgba = cv2.split(image_np)
    alpha = rgba[3]

    # 使用阈值将非透明部分转换为纯白色（255），透明部分转换为纯黑色（0）
    _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取最大轮廓的边界框
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # 使用边界框裁剪图像
    cropped_image = image_np[y:y+h, x:x+w]

    # 将裁剪后的图像转换为PIL的Image类型
    result_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2RGBA))

    return result_pil






# result_image=process_image('1.png',-12,0.1)

image = Image.open('2.png')
result_image=crop_image_remove_transparent(image)

result_image.save('test.png')



