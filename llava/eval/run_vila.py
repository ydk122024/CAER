# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
import torch
from PIL import Image

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args):
    """
    评估模型性能的函数。
    
    根据提供的参数，该函数将加载图像或视频，预处理图像，然后使用预训练的模型进行生成对话。
    参数:
    - args: 包含模型路径、查询字符串、视频文件路径等的参数对象。
    
    返回:
    - outputs: 模型生成的输出字符串。
    """
    # 禁用PyTorch的默认初始化，可能为了使用特定的初始化方法。
    # Model
    disable_torch_init()
    # 根据是否有视频文件，加载图像。
    if args.video_file is None:
        # 解析图像文件路径，加载图像。
        image_files = image_parser(args)
        images = load_images(image_files)
    else:
        # 从URL下载视频或从本地路径加载视频。
        if args.video_file.startswith("http") or args.video_file.startswith("https"):
            print("downloading video from url", args.video_file)
            response = requests.get(args.video_file)
            video_file = BytesIO(response.content)
        else:
            assert osp.exists(args.video_file), "video file not found"
            video_file = args.video_file
        # 从视频中提取帧。
        from llava.mm_utils import opencv_extract_frames
        images = opencv_extract_frames(video_file, args.num_video_frames)
        
    # 加载预训练模型及其相关组件。
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    # 处理查询字符串，根据模型配置插入图像标记。
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("no <image> tag found in input. Automatically append one at the beginning of text.")
            # do not repeatively append the prompt.
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
    # 打印处理后的输入。
    print("input: ", qs)

    # 根据模型名称确定对话模式。
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 检查命令行参数指定的对话模式是否与自动推断的模式一致。
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # 初始化对话模板并添加输入查询。
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 预处理图像，准备用于模型输入。
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    # 准备模型输入的prompt令牌。
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # 根据对话模式确定停止标记。
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    # 设置停止生成的条件。
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # 打印预处理图像的形状。
    print(images_tensor.shape)
    # 使用推理模式以优化性能。
    with torch.inference_mode():
        # 生成模型输出。
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # 解码模型输出，移除特殊令牌。
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # 移除对话模式的停止标记。
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # 打印最终输出。
    print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-2.7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)