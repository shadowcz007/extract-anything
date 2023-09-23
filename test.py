from pymatting import *
import numpy as np
from PIL import Image
import io

scale = 1.0



def image_open_to_png(filepath):
    with Image.open(filepath) as img:
        # 创建一个内存缓冲区
        buffer = io.BytesIO()
        
        # 将图像保存到内存缓冲区中，格式为PNG
        img.save(buffer, format="PNG")
        
        # 将内存缓冲区的内容转换为字节数据
        png_bytes = buffer.getvalue()
        image_file = io.BytesIO(png_bytes)
        # Image.open(image_file)
        # 使用 Image.open 打开文件对象，返回图像对象
        return image_file


image = load_image(image_open_to_png("./data/154149295621809200_w_1198x800.jpg"), "RGB", scale, "box")
trimap = load_image(image_open_to_png("./data/image.png"), "GRAY", scale, "nearest")
# print(trimap)
# estimate alpha from image and trimap
# alpha = estimate_alpha_cf(image, trimap)
# alpha = estimate_alpha_knn(
#     image,
#     trimap,
#     laplacian_kwargs={"n_neighbors": [15, 10]},
#     cg_kwargs={"maxiter":2000})
alpha = estimate_alpha_cf(
    image,
    trimap,
    laplacian_kwargs={"epsilon": 1e-6},
    cg_kwargs={"maxiter":2000})

# make gray background
background = np.zeros(image.shape)
background[:, :] = [0.5, 0.5, 0.5]

# estimate foreground from image and alpha
foreground = estimate_foreground_ml(image, alpha)

# blend foreground with background and alpha, less color bleeding
new_image = blend(foreground, background, alpha)

# save results in a grid
images = [image, trimap, alpha, new_image]
grid = make_grid(images)
save_image("./data/lemur_grid.png", grid)

# save cutout
cutout = stack_images(foreground, alpha)
save_image("./data/lemur_cutout.png", cutout)

# just blending the image with alpha results in color bleeding
color_bleeding = blend(image, background, alpha)
grid = make_grid([color_bleeding, new_image])
save_image("./data/lemur_color_bleeding.png", grid)