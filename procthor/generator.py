import os
import numpy as np
from tqdm import tqdm
from ai2thor.controller import Controller
import random
from PIL import Image
from action import *
import cv2
import prior
import pickle
import torch
import csv
import argparse
import time
import pdb


def parse_arguments():
    parser = argparse.ArgumentParser('Parse main configuration file', add_help=False)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--save_dir", default='./data', type=str)
    parser.add_argument("--sample_per_scene", default=20, type=int)
    parser.add_argument("--min_scene_idx", default=0, type=int, help='min scene idx')
    parser.add_argument("--max_scene_idx", default=9999, type=int, help='min scene idx')
    parser.add_argument("--save_bbox", default=False, action='store_true')
    return parser.parse_args()


class MultiObjEnv:
    def __init__(self, save_dir="./data", save_bbox=False, width=672, height=672, local_executable_path=None):
        self.save_dir = save_dir
        self.width = width
        self.height = height
        self.x_display = "0"
        self.actionable = get_actionable()
        self.blacklist = ['StoveKnob', 'Toaster', 'CoffeeMachine', 'Kettle', 'Doorway']
        self.save_bbox = save_bbox

        self.controller = Controller(
            branch="nanna",
            scene="Procedural",
            visibilityDistance=5.0,

            # # step sizes
            # gridSize=0.4,
            # snapToGrid=True,
            # rotateStepDegrees=90,

            # image modalities
            renderDepthImage=False,
            renderInstanceSegmentation=True,

            # camera properties
            width=672,
            height=672,
            fieldOfView=90
        )

        self.dataset = prior.load_dataset("procthor-10k")

    def _create_folders(self):

        self.dircolor = os.path.join(self.save_dir, self.scene, 'color')
        if not os.path.exists(self.dircolor):
            os.makedirs(self.dircolor)

        self.dirbbox = os.path.join(self.save_dir, self.scene, 'bbox')
        if not os.path.exists(self.dirbbox):
            os.makedirs(self.dirbbox)

        self.dirmask = os.path.join(self.save_dir, self.scene, 'mask')
        if not os.path.exists(self.dirmask):
            os.makedirs(self.dirmask)

        self.csvname = os.path.join(self.save_dir, self.scene, 'annotations.csv')

    def _init_appearance(self, seed=0):
        self.controller.step(
            action="RandomizeMaterials"
        )

        self.controller.step(
            action="RandomizeLighting",
            brightness=(1.0, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False
        )

    def _set_state(self, position, rotation):
        # print("Teleporting the agent to", position, " with rotation", rotation)
        self.state = self.controller.step(action="Teleport", position=position, rotation=rotation)

    def _check_properties(self, obj):
        properties = list()
        for adj in self.actionable.keys():
            if obj[adj]:
                properties.append(adj)
        return properties

    def _check_objects(self):
        objects = self.state.metadata['objects']
        objects = [obj for obj in objects if obj['visible']]
        stack = list()
        for obj in objects:
            if obj['objectType'] in self.blacklist:
                continue
            adj = self._check_properties(obj)
            # if adj:
            #     print(adj)
            # pdb.set_trace()
            if adj and obj['objectId'] in self.state.instance_detections2D.instance_masks.keys():
                (xmin, ymin, xmax, ymax) = self.state.instance_detections2D[obj['objectId']]
                if (xmax - xmin) < 60 or (ymax - ymin) < 60:
                    continue
                if xmin > 20 and ymin > 20 and xmax < self.width - 20 and ymax < self.height - 20:
                    # object fully in the field of view
                    stack.append((obj, adj))
                elif (xmax - xmin) > 100 and (ymax - ymin) > 100:
                    # object largely in the field of view
                    stack.append((obj, adj))
                else:
                    print('boader object', obj['objectType'], xmin, ymin, xmax, ymax)

        return stack

    def _sample_intervention(self, items):
        (obj, adj) = random.choice(items)
        (key, [act_1, act_2]) = self.actionable[random.choice(adj)]
        if obj[key]:
            return obj, act_1
        else:
            return obj, act_2

    def _mask_image(self, mask):
        img = np.zeros(mask.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j]:
                    img[i][j] = 255
        return img

    def _collect_state(self, seed, max_samples=5, min_obj=2, min_change=1000, epsilon=20):
        self._init_appearance(seed)
        count = 0
        if self.reachable_positions:
            random.shuffle(self.reachable_positions)
        else:
            return 0
        for position in self.reachable_positions:
            rotation = random.choice(range(360))
            self._set_state(position, rotation)
            items = self._check_objects()
            if len(items) >= min_obj:
                obj, intervention = self._sample_intervention(items)
                if intervention:
                    noun = obj['objectType']
                    verb = intervention.__name__[5:]
                    color_before = self.state.frame
                    bbox_before = self.state.instance_detections2D[obj['objectId']]
                    mask_before = self.state.instance_masks[obj['objectId']]

                    # intervention
                    self.state = intervention(self.controller, obj['objectId'])
                    color_after = self.state.frame
                    if obj['objectId'] in self.state.instance_detections2D.instance_masks.keys():
                        bbox_after = self.state.instance_detections2D[obj['objectId']]
                        mask_after = self.state.instance_masks[obj['objectId']]
                    else:
                        continue

                    # bbox IOU
                    xmin = max(bbox_before[0], bbox_after[0])
                    ymin = max(bbox_before[1], bbox_after[1])
                    xmax = min(bbox_before[2], bbox_after[2])
                    ymax = min(bbox_before[3], bbox_after[3])
                    change_in_box = (abs(color_after[ymin:ymax+1, xmin:xmax+1] - color_before[ymin:ymax+1, xmin:xmax+1]) > 0).max(axis=2).sum()

                    xmin = min(bbox_before[0], bbox_after[0])
                    ymin = min(bbox_before[1], bbox_after[1])
                    xmax = max(bbox_before[2], bbox_after[2])
                    ymax = max(bbox_before[3], bbox_after[3])
                    change_total = (abs(color_after - color_before) > 0).max(axis=2).sum()
                    change_out_box = change_total - (abs(color_after[ymin:ymax+1, xmin:xmax+1] - color_before[ymin:ymax+1, xmin:xmax+1]) > 0).max(axis=2).sum()

                    if change_in_box > min_change and change_out_box < epsilon:   # epsilon
                        count += 1
                        self.cnt_data += 1
                        figname_before = f'{self.cnt_data:05d}_first'
                        figname_after = f'{self.cnt_data:05d}_second_{noun}_{verb}'
                        self._save_img(color_before, bbox_before, mask_before, figname_before)
                        self._save_annotation(figname_before, noun, 'none', bbox_before)
                        self._save_img(color_after, bbox_after, mask_after, figname_after)
                        self._save_annotation(figname_after, noun, verb, bbox_after)
            if count >= max_samples:    # only sample a few intervened pairs per scene
                return count
        return count

    def _save_img(self, color, bbox, mask, figname='temp'):
        Image.fromarray(color).save(os.path.join(self.dircolor, figname+".png"), "png")

        if self.save_bbox:
            (xmin, ymin, xmax, ymax) = bbox
            color = color.copy()
            color[ymin:ymin+4, xmin:xmax] = (255, 255, 255)
            color[ymax-4:ymax, xmin:xmax] = (255, 255, 255)
            color[ymin:ymax, xmin:xmin+4] = (255, 255, 255)
            color[ymin:ymax, xmax-4:xmax] = (255, 255, 255)
            Image.fromarray(color).save(os.path.join(self.dirbbox, figname+".png"), "png")

        mask = self._mask_image(mask)
        cv2.imwrite(os.path.join(self.dirmask, figname+".png"), mask)

    def _save_annotation(self, figname, noun, verb, bbox):
        row = [self.scene, self.cnt_data, figname, noun, verb, bbox[0], bbox[1], bbox[2], bbox[3]]
        # open the file in the write mode
        with open(self.csvname, 'a', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(row)

    def reset_house(self, idx, seed=0):
        house = self.dataset["train"][idx]
        self.scene = f'proc_{idx:05d}'
        self._create_folders()
        self.controller.reset()
        self.controller.step(action="CreateHouse", house=house)

        self.state = self.controller.step(
            action="InitialRandomSpawn",
            randomSeed=seed,
            forceVisible=True,
            numPlacementAttempts=5,
            placeStationary=True
        )

        self.reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        # print("Found #", len(self.reachable_positions), "reachable positions.")

    def generate_data(self, scene_idx, num_data):
        self.cnt_data = 0
        self.annotations = list()
        for seed in range(int(num_data/3)):
            # print(f"[INFO] ready to collect data #{seed}")
            self.reset_house(scene_idx, seed)
            self._collect_state(seed)
            if self.cnt_data > num_data:
                break


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    env = MultiObjEnv(save_dir=args.save_dir, save_bbox=args.save_bbox)

    tic = time.time()
    for i in range(args.min_scene_idx, args.max_scene_idx+1):
        env.generate_data(i, args.sample_per_scene)
        toc = time.time()
        print(f"Scene {i} ({args.min_scene_idx} - {args.max_scene_idx}), elapsed {(toc-tic)/60.0:.1f} min")
        tic = toc
