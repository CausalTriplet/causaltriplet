# built from https://github.com/facebookresearch/segment-anything

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import os

from segment_anything import sam_model_registry, SamPredictor

import pdb

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--px",
    type=int,
    required=True,
    help="The x coordinate of the point prompt.",
)

parser.add_argument(
    "--py",
    type=int,
    required=True,
    help="The y coordinate of the point prompt.",
)


parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=175):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)

    # raw images
    image = cv2.imread(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # prompt point
    input_point = np.array([[args.px, args.py]])
    input_label = np.array([1])

    print('Prompt: ', input_point)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    plt.figure(figsize=(10,10))
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')

        base = os.path.basename(args.input)
        base = os.path.splitext(base)[0]
        figname = base + f'_mask_{i}.png'
        plt.savefig(os.path.join(args.output, figname), bbox_inches='tight', pad_inches=0)
        print(f'Mask #{i} is saved to {os.path.join(args.output, figname)}')

        plt.clf()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
