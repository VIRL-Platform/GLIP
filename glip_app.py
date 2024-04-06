"""
Modified from https://huggingface.co/spaces/haotiz/glip-zeroshot-demo/blob/main/app.py
Modified by Runyu DING
"""

# import requests
# import os
# from io import BytesIO
# from PIL import Image
# import numpy as np
# from pathlib import Path
import gradio as gr
import argparse

import torch

import warnings

warnings.filterwarnings("ignore")

# os.system(
#     "pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo click opencv-python inflect nltk scipy scikit-learn pycocotools")
# os.system("pip install transformers")
# os.system("python setup.py build develop --user")

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from glip_demo import GLIPModel


parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--model_size", default="tiny")
parser.add_argument("--root_path", default="./")
# parser.add_argument("--score_thresh", default=0.7)

# Use this command if you want to try the GLIP-L model
# ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
# config_file = "configs/pretrain/glip_Swin_L.yaml"
# weight_file = "MODEL/glip_large_model.pth"

# update the config options with the config file
# manual override some options
parser.add_argument("--local_rank", default=0)
parser.add_argument("--num_gpus", default=1)
args = parser.parse_args()

# cfg.merge_from_file(args.config_file)
# cfg.merge_from_list(["model_size", args.model_size])
# cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


glip_model = GLIPModel(
    args, "cuda" if torch.cuda.is_available() else "cpu"
)


# def predict(image, text):
#     result, _ = glip_demo.run_on_web_image(image[:, :, [2, 1, 0]], text, 0.5)
#     return result[:, :, [2, 1, 0]]

# ======== input =========
image_input = gr.Image(type="pil")

caption_input = gr.Textbox(label="Caption:", placeholder="prompt", lines=2)

score_thresh = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.1,
                interactive=True,
                label="Score Threshold",
            )

need_draw = gr.Checkbox(label="Draw", info="Do you wan to draw the results?")

# ======== ouptut =============
text_output = gr.JSON(label="Output text")

image_output = gr.outputs.Image(
            type="pil",
            label="grounding results"
        )


gr.Interface(
    description="Object Detection in the Wild through GLIP (https://github.com/microsoft/GLIP).",
    fn=glip_model.inference,
    inputs=[image_input, caption_input, score_thresh, need_draw],
    outputs=[text_output, image_output]
).launch(share=True, enable_queue=True, server_port=7862)
# ).launch(server_name="0.0.0.0", server_port=7000, share=True)
