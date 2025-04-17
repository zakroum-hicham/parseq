#!/usr/bin/env python
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string
import sys
import os

### ZAK
ROOT = os.path.join(os.getcwd(),"str","parseq")
###
sys.path.append(str(ROOT))  # add ROOT to PATH


import torch

from tqdm import tqdm

from model.help import load_from_checkpoint, parse_model_args

from PIL import Image
import json

from typing import Tuple
from torchvision import transforms as T

def get_transform(img_size: Tuple[int]):
        transforms = []
        transforms.extend([
            T.Resize(img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        return T.Compose(transforms)

def run_inference(model, data_root, result_file, img_size):
    # load images one by one, save paths and result
    file_dir = os.path.join(data_root, 'imgs')
    filenames = os.listdir(file_dir)
    filenames.sort()
    results = {}
    for filename in tqdm(filenames):
        image = Image.open(os.path.join(file_dir, filename)).convert('RGB')
        transform = get_transform(img_size)
        image = transform(image)
        image = image.unsqueeze(0)
        logits = model.forward(image.to(model.device))
        #convert to 3 by 10
        probs_full = logits[:,:3,:11].softmax(-1)
        preds, probs = model.tokenizer.decode(probs_full)
        logits = logits[:,:3,:11].cpu().detach().numpy()[0].tolist()
        # probs = logits.softmax(-1)
        # preds, probs = model.tokenizer.decode(probs)
        probs_full = probs_full.cpu().detach().numpy()[0].tolist()
        confidence = probs[0].cpu().detach().numpy().squeeze().tolist()
        ### ZAK
        results[filename] = {'label':preds[0], 'confidence':confidence}
        ### 
    with open(result_file, 'w') as f:
        json.dump(results, f)



@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--result_file', default='outputs/preds.json')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(kwargs)

    charset_test = string.digits # + string.ascii_lowercase

    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    run_inference(model, args.data_root, args.result_file, hp.img_size)


if __name__ == '__main__':
    main()