import argparse
import string
import sys
import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Tuple
from torchvision import transforms as T
import torch

### ZAK
ROOT = os.path.join(os.getcwd(), "str", "parseq")
sys.path.append(str(ROOT))  # add ROOT to PATH

from model.help import load_from_checkpoint, parse_model_args


def get_transform(img_size: Tuple[int]):
    transforms = [
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ]
    return T.Compose(transforms)


def run_inference_on_tracklets(model, data_root, result_file, img_size, batch_size):
    imgs_root = Path(data_root) / "imgs"
    transform = get_transform(img_size)
    results = {}

    # Collect image paths grouped by tracklet ID
    all_samples = []
    for tracklet_folder in sorted(imgs_root.iterdir()):
        if not tracklet_folder.is_dir():
            continue
        tracklet_id = tracklet_folder.name
        for img_file in sorted(tracklet_folder.glob("*.jpg")):
            all_samples.append((tracklet_id, img_file))

    # Process in batches
    for i in tqdm(range(0, len(all_samples), batch_size)):
        batch_samples = all_samples[i:i+batch_size]
        images = []
        meta = []

        for tracklet_id, img_path in batch_samples:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image)
            images.append(image_tensor)
            meta.append((tracklet_id, img_path.name))

        images_tensor = torch.stack(images).to(model.device)
        logits = model.forward(images_tensor)
        probs_full = logits[:, :3, :11].softmax(-1)
        preds, probs = model.tokenizer.decode(probs_full)

        for j, (tracklet_id, filename) in enumerate(meta):
            confidence = probs[j].cpu().detach().numpy().squeeze().tolist()
            if tracklet_id not in results:
                results[tracklet_id] = []
            results[tracklet_id].append({
                "filename": filename,
                "label": preds[j],
                "confidence": confidence
            })

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--result_file', default='outputs/tracklet_preds.json')
    parser.add_argument('--batch_size', default=16)
    args, unknown = parser.parse_known_args()

    kwargs = parse_model_args(unknown)
    charset_test = string.digits  # Or add letters if needed
    kwargs.update({'charset_test': charset_test})

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    run_inference_on_tracklets(model, args.data_root, args.result_file, hp.img_size, int(args.batch_size))


if __name__ == '__main__':
    main()
