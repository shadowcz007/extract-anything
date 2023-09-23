
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

import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

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

config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
# sam_checkpoint = './sam_vit_h_4b8939.pth' 
sam_checkpoint = './checkpoints/sam_vit_h_4b8939.pth' 
output_dir = "outputs"
device = 'cpu'

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
ram_model = None
kosmos_model = None
kosmos_processor = None

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_image", "-i", type=str, default="", help="")
    argparser.add_argument("--text", "-t", type=str, default="", help="")
    argparser.add_argument("--output_image", "-o", type=str, default="", help="")
    args = argparser.parse_args()
    return args

# usage: 
#       python app_cli.py --input_image dog.png --text dog --output_image dog_remove.png

if __name__ == '__main__':
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

    input_image = Image.open(args.input_image)

    output_images, _ = run_anything_task(input_image = input_image, 
                        text_prompt = args.text,  
                        task_type = 'remove', 
                        inpaint_prompt = '', 
                        box_threshold = 0.3, 
                        text_threshold = 0.25, 
                        iou_threshold = 0.8, 
                        inpaint_mode = "merge", 
                        mask_source_radio = "type what to detect below", 
                        remove_mode = "rectangle",   # ["segment", "rectangle"]
                        remove_mask_extend = "10", 
                        num_relation = 5,
                        kosmos_input = None,
                        cleaner_size_limit = -1,
                        )
    if len(output_images) > 0:
        logger.info(f'save result to {args.output_image} ... ')        
        output_images[-1].save(args.output_image)
        # count = 0
        # for output_image in output_images:
        #     count += 1
        #     if isinstance(output_image, np.ndarray):
        #         output_image = PIL.Image.fromarray(output_image.astype(np.uint8))
        #     output_image.save(args.output_image.replace(".",  f"_{count}."))
