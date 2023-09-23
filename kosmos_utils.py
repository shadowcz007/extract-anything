import random
import numpy as np
import os
import requests
import torch
import torchvision.transforms as torchvision_T
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import cv2
import ast

colors = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
    (255, 0, 0),    
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for color_id, color in enumerate(colors)
}


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def draw_entity_boxes_on_image(image, entities, show=False, save_path=None, entity_index=-1):
    """_summary_
    Args:
        image (_type_): image or image path
        collect_entity_location (_type_): _description_
    """
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        # pdb.set_trace()
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = torchvision_T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")
    
    if len(entities) == 0:
        return image

    indices = list(range(len(entities)))
    if entity_index >= 0:
        indices = [entity_index]

    # Not to show too many bboxes
    entities = entities[:len(color_map)]
    
    new_image = image.copy()
    previous_bboxes = []
    # size of text
    text_size = 1
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 3
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    # num_bboxes = sum(len(x[-1]) for x in entities)
    used_colors = colors  # random.sample(colors, k=num_bboxes)

    color_id = -1
    for entity_idx, (entity_name, (start, end), bboxes) in enumerate(entities):
        color_id += 1
        if entity_idx not in indices:
            continue
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
            # if start is None and bbox_id > 0:
            #     color_id += 1
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)

            # draw bbox
            # random color
            color = used_colors[color_id]  # tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

            x1 = orig_x1 - l_o
            y1 = orig_y1 - l_o

            if y1 < text_height + text_offset_original + 2 * text_spaces:
                y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                x1 = orig_x1 + r_o

            # add text background
            (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

            for prev_bbox in previous_bboxes:
                while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                    text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                    text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                    y1 += (text_height + text_offset_original + 2 * text_spaces)

                    if text_bg_y2 >= image_h:
                        text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                        text_bg_y2 = image_h
                        y1 = image_h
                        break

            alpha = 0.5
            for i in range(text_bg_y1, text_bg_y2):
                for j in range(text_bg_x1, text_bg_x2):
                    if i < image_h and j < image_w:
                        if j < text_bg_x1 + 1.35 * c_width:
                            # original color
                            bg_color = color
                        else:
                            # white
                            bg_color = [255, 255, 255]
                        new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

            cv2.putText(
                new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
            )
            # previous_locations.append((x1, y1))
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

    pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
    if save_path:
        pil_image.save(save_path)
    if show:
        pil_image.show()

    return pil_image

def load_kosmos_model(device):
    ckpt = "ydshieh/kosmos-2-patch14-224"
    kosmos_model = AutoModelForVision2Seq.from_pretrained(ckpt, trust_remote_code=True).to(device)
    kosmos_processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    return kosmos_model, kosmos_processor

def kosmos_generate_predictions(image_input, text_input, kosmos_model, kosmos_processor):
    if kosmos_model is None:
        return None, None, None

    # Save the image and load it again to match the original Kosmos-2 demo.
    # (https://github.com/microsoft/unilm/blob/f4695ed0244a275201fff00bee495f76670fbe70/kosmos-2/demo/gradio_app.py#L345-L346)
    user_image_path = "/tmp/user_input_test_image.jpg"
    image_input.save(user_image_path)
    # This might give different results from the original argument `image_input`
    image_input = Image.open(user_image_path)

    if text_input == "Brief":
        text_input = "<grounding>An image of"
    elif text_input == "Detailed":
        text_input = "<grounding>Describe this image in detail:"
    else:
        text_input = f"<grounding>{text_input}"

    inputs = kosmos_processor(text=text_input, images=image_input, return_tensors="pt")

    generated_ids = kosmos_model.generate(
        pixel_values=inputs["pixel_values"].to("cuda"),
        input_ids=inputs["input_ids"][:, :-1].to("cuda"),
        attention_mask=inputs["attention_mask"][:, :-1].to("cuda"),
        img_features=None,
        img_attn_mask=inputs["img_attn_mask"][:, :-1].to("cuda"),
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = kosmos_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # By default, the generated  text is cleanup and the entities are extracted.
    processed_text, entities = kosmos_processor.post_process_generation(generated_text)

    annotated_image = draw_entity_boxes_on_image(image_input, entities, show=False)

    color_id = -1
    entity_info = []
    filtered_entities = []
    for entity in entities:
        entity_name, (start, end), bboxes = entity
        if start == end:
            # skip bounding bbox without a `phrase` associated
            continue
        color_id += 1
        # for bbox_id, _ in enumerate(bboxes):
            # if start is None and bbox_id > 0:
            #     color_id += 1
        entity_info.append(((start, end), color_id))
        filtered_entities.append(entity)

    colored_text = []
    prev_start = 0
    end = 0
    for idx, ((start, end), color_id) in enumerate(entity_info):
        if start > prev_start:
            colored_text.append((processed_text[prev_start:start], None))
        colored_text.append((processed_text[start:end], f"{color_id}"))
        prev_start = end

    if end < len(processed_text):
        colored_text.append((processed_text[end:len(processed_text)], None))

    return annotated_image, colored_text, str(filtered_entities)