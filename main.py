
import warnings
warnings.filterwarnings('ignore')

import subprocess, io, os, sys, time
from loguru import logger

# os.system("pip install diffuser==0.6.0")
# os.system("pip install transformers==4.29.1")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if os.environ.get('IS_MY_DEBUG') is None:
    result = subprocess.run(['pip', 'install', '-e', 'GroundingDINO'], check=True)
    print(f'pip install GroundingDINO = {result}')

# result = subprocess.run(['pip', 'list'], check=True)
# print(f'pip list = {result}')

sys.path.insert(0, './GroundingDINO')

import gradio as gr

import argparse

import copy,base64

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import cv2
import numpy as np
import matplotlib.pyplot as plt
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config as lama_Config

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download

from  utils import computer_info
# relate anything
from ram_utils import iou, sort_and_deduplicate, relation_classes, MLP, show_anns, ram_show_mask
from ram_train_eval import RamModel,RamPredictor
from mmengine.config import Config as mmengine_Config
from app import *

from fastapi import FastAPI, UploadFile, File,Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse


config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
# sam_checkpoint = './sam_vit_h_4b8939.pth' 
sam_checkpoint = './checkpoints/sam_vit_h_4b8939.pth' 
output_dir = "outputs"
device = 'cuda'

# lama的模型存储位置,保存到项目所在
os.environ['TORCH_HOME']="./checkpoints"

os.makedirs(output_dir, exist_ok=True)
groundingdino_model = None
sam_device = None
sam_model = None
sam_predictor = None
sam_mask_generator = None
sd_pipe = None
lama_cleaner_model= None
lama_cleaner_model_device = device
ram_model = None
kosmos_model = None
kosmos_processor = None



def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_image", "-i", type=str, default="", help="")
    argparser.add_argument("--text", "-t", type=str, default="anything,human,person,logo,object", help="")
    argparser.add_argument("--output_image", "-o", type=str, default="test.png", help="")
    args = argparser.parse_args()
    return args



# usage: 
#       python app_cli.py --input_image dog.png --text dog --output_image dog_remove.png

dir_path='./temp'
if not os.path.exists(dir_path):
    # 目录不存在，创建目录
    os.makedirs(dir_path)
    print(f"目录 '{dir_path}' 创建成功！")
else:
    print(f"目录 '{dir_path}' 已经存在！")

def mask_image(im1p,im2p,resp):
    print(im1p,im2p,resp)
    
    # Load two images
    im1 = cv2.cvtColor(np.array(im1p), cv2.COLOR_RGB2BGR)
    im2 = cv2.cvtColor(np.array(im2p), cv2.COLOR_RGB2BGR)
    # im1 = cv2.imread(im1p) # 背景
    # im2 = cv2.imread(im2p) # logo

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = im2.shape
    roi = im1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    im2gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(im2gray, 254, 255, cv2.THRESH_BINARY) # 这个254很重要
    mask_inv = cv2.bitwise_not(mask)

    # Convert the mask to 3 channels
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

    # cv.imshow('mask',mask_inv)
    # Now black-out the area of logo in ROI
    im1_bg = cv2.bitwise_and(roi,roi,mask = mask) 

    # Take only region of logo from logo image.
    im2_fg = cv2.bitwise_and(im2,im2,mask = mask_inv) 

    # Put logo in ROI and modify the main image
    dst = cv2.add(im1_bg,im2_fg)
    im1[0:rows, 0:cols ] = dst

    image_rgba = cv2.cvtColor(im1, cv2.COLOR_BGR2BGRA)
    # 创建一个与图像大小相同的透明图像
    transparent_image = np.zeros_like(image_rgba)

    # 使用掩码将黑色背景转换为透明
    mask1 = np.all(im1[:, :, :3] == [0, 0, 0], axis=-1)
    transparent_image[mask1] = [0, 0, 0, 0]
    transparent_image[~mask1] = image_rgba[~mask1]

    # Save the result with transparent background as PNG
    # 将图像转换为Base64编码
    ret, buffer = cv2.imencode('.png', transparent_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # cv2.imwrite(resp, im1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # cv.imshow('res',im1)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return "data:image/png;base64," + base64_image


def has_transparency(image):
    # image = Image.open(image_path)
    if image.mode == 'RGBA':
        alpha = image.split()[3]
        return alpha.getbbox() is not None
    return False

# 调用示例
# image_path = 'path_to_your_image.png'
# if has_transparency(image_path):
#     print("图片具有透明底")
# else:
#     print("图片没有透明底")



# 黑色为孔洞和背景
def fill(img, num_fill=1):
    img = img.convert('L')
    img = np.array(img)
    
    # Find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea,reverse=True)
    
    # Initialize output image
    out_img = np.zeros((img.shape[0], img.shape[1]))
    
    # Fill contours
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        print("Contour area:", area)
        
        cv2.fillPoly(out_img, [cnt], color=255 if i<min(num_fill,len(contours) ) else 0)
    
    return Image.fromarray(out_img).convert('L')


def process_image(image, start_offset=18, feathering_weight=0.8):
    # Open the image using PIL
    image =image.convert("L")
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


def im_to_base64(im):
    # print('####', im)
    # 转换为字节流
    image_byte = BytesIO()
    im.save(image_byte, format="PNG")
    image_byte = image_byte.getvalue()
    # 转换为Base64编码
    image_base64 = base64.b64encode(image_byte).decode("utf-8")
    # print('####',image_base64)
    f="png"
    if im.format!=None:
        f=im.format.lower()
    return "data:image/" + f + ";base64," + image_base64


def blur(image,radius=2):

    # radius = 2

    #创建平滑滤波器对象
    smooth_filter = ImageFilter.SMOOTH_MORE

    #调整滤波器的幅度
    smooth_filter = smooth_filter(radius)

    #应用平滑滤波器
    smoothed_image = image.filter(smooth_filter)
 
    return smoothed_image


app = FastAPI()

# 添加跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return HTMLResponse("""
        <html>
                        <style>
                        img{
                            background:'gray';
                            width: 300px;
                            margin: 24px;}
                        </style>
<body>
<form id="myForm" enctype="multipart/form-data">
<input name="file" type="file">
                        <label for='prompt'>prompt</label>
<input name="prompt" type="text" value="anything,human,person,logo,object,fruit">
                        <label for='margin'>margin</label>
<input name="margin" type="number" value="12">
                        <label for='fill_num'>fill_num</label>
                        <input name="fill_num" type="number" value="2">
                        <label for='blur_num'>blur_num</label>
                        <input name="blur_num" type="number" value="2">
<input type="button" value="Submit" onclick="submitForm()">
</form>

    <div id="result"></div>

    <script>
        const createImg=(url)=>{
                    var im=new Image()
                    im.src=url;
                    document.querySelector('#result').appendChild(im)
        }
        function submitForm() {
            var form = document.getElementById("myForm");
            var formData = new FormData(form);

            fetch("/upload", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                var result = document.getElementById("result");
                        result.innerHTML='';
                        createImg(data['origin'])
                createImg(data['1'])
                 createImg(data['4'])
                        createImg(data['5'])
                        createImg(data['6'])
                        createImg(data['7'])
                         createImg(data['8'])
                        createImg(data['result'])
            })
            .catch(error => {
                console.error("Error:", error);
            });
        }
    </script>
</body>
</html>
    """)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), 
                      prompt: str = Form(...),
                      margin: int = Form(...),
                      fill_num: int = Form(...),
                      blur_num:int=Form(...)
                      ):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    if has_transparency(image):
        print("图片具有透明底")
        mask = image.split()[3] # 获取透明通道
        mask=mask.convert("RGB")
        image = image.convert("RGB") # 转换为RGB模式
    else:
        print("图片没有透明底")
        mask=image.convert("RGB")
        image=image.convert("RGB")


    # 在这里添加对上传图片的处理逻辑
    # 返回处理结果，包括Base64编码的图片
    input_image = {'image':image,'mask':mask}

    # 需要定义prompt来实现更好的抠图效果
    text=prompt
    if prompt==None or prompt == "":
        text='anything,human,person,logo,object,fruit'

    

    result = run_anything_task(input_image = input_image, 
                            text_prompt =text,  
                            task_type = 'remove', 
                            inpaint_prompt = '', 
                            box_threshold = 0.3, 
                            text_threshold = 0.25, 
                            iou_threshold = 0.8, 
                            inpaint_mode = "merge", 
                            mask_source_radio = "type what to detect below", 
                            remove_mode = "segment",   # ["segment", "rectangle"]
                            remove_mask_extend = "10", 
                            num_relation = 5,
                            kosmos_input = None,
                            cleaner_size_limit = -1,
                            )
    # print(result)
    images=result[0]
    print(len(images))
    # 返回处理结果
    if len(images)>0:
        im5 = ImageOps.invert(images[4])
        fill_mask=fill(images[4],fill_num)
        return {
            "filename": file.filename, 
            "origin":im_to_base64(images[0]),
            "1":im_to_base64(images[1]),#识别出的物体区域
            "4":im_to_base64(images[4]),#遮罩图:白色为物体
            "5":im_to_base64(im5),#反转图：黑色为物体,
            "6":im_to_base64(process_image(images[4],margin,1)),#向外扩充18
            "7":im_to_base64(process_image(images[4],-margin,1)),#向内缩减18
            "8":im_to_base64(fill_mask),#填充较小的孔洞,传入数量可以控制填充孔洞的数量
            "9":im_to_base64(blur(fill_mask,blur_num)),
            "result":mask_image(images[0],fill_mask,"")
            }
    else:
        {
            "filename": file.filename,            
            }



if __name__ == '__main__':
    import uvicorn

    args = get_args()
    logger.info(f'\nargs={args}\n')

    logger.info(f'loading models ... ')
    # set_device()  # If you have enough GPUs, you can open this comment
    # get_sam_vit_h_4b8939()
    load_groundingdino_model()
    load_sam_model()
    # load_sd_model()
    load_lama_cleaner_model()
    # load_ram_model()

    uvicorn.run(app, host="0.0.0.0", port=3033)
    # input_image = {'image':Image.open(args.input_image),'mask':Image.open(args.input_image)}

    # input_image={'image': <PIL.Image.Image image mode=RGB size=512x512 at 0x157B8B38760>, 
    #  'mask': <PIL.Image.Image image mode=RGB size=512x512 at 0x157B8B3A410>} 
    
    # text= anything
    
    # task_type=remove

    # box_threshold=0.3 
    
    # text_threshold=0.25 
    
    # iou_threshold=0.8 
    
    # inpaint_mode=merge 
    
    # mask_source_radio=type what to detect below 

    # remove_mode=segment 

    # remove_mask_extend=10 
    
    # num_relation=5 

    # kosmos_input=Brief 
    
    # cleaner_size_limit=1080

    # result = run_anything_task(input_image = input_image, 
    #                     text_prompt = args.text,  
    #                     task_type = 'remove', 
    #                     inpaint_prompt = '', 
    #                     box_threshold = 0.3, 
    #                     text_threshold = 0.25, 
    #                     iou_threshold = 0.8, 
    #                     inpaint_mode = "merge", 
    #                     mask_source_radio = "type what to detect below", 
    #                     remove_mode = "segment",   # ["segment", "rectangle"]
    #                     remove_mask_extend = "10", 
    #                     num_relation = 5,
    #                     kosmos_input = None,
    #                     cleaner_size_limit = -1,
    #                     )
   
    # output_images=result[0]
   
    # if len(output_images) > 0:
    #     # logger.info(f'save result to {args.output_image} ... ')        
    #     # output_images[-1].save(args.output_image)
      
    #     count = 0
    #     for im in output_images:
    #         count += 1
    #         if isinstance(im, np.ndarray):
    #             output_image = PIL.Image.fromarray(im.astype(np.uint8))
    #             fn=args.output_image.replace(".",  f"_{count}.")
    #             output_image.save('temp/'+fn)

    #     mask_image('temp/'+args.output_image.replace(".",  f"_{1}."),
    #     'temp/'+args.output_image.replace(".",  f"_{4}."),
    #     args.output_image)
