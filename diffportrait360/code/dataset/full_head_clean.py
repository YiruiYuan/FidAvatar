import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
print("using load_truncated_images, to avoid bug")


def read_image_512(path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((512, 512))
    return np.array(image)


class back_head_generation(Dataset):
    def __init__(self, inference_image_dataset, condition_path, image_transform=None, initial_image_path=None, use_initial_condition=False):
        self.root = inference_image_dataset
        self.condition_path = condition_path
        self.initial_image_path = initial_image_path
        self.use_initial_condition = use_initial_condition
        self.condition_frame_idx = 18
        self.transform = image_transform
        self.id_list = [
            file_name
            for file_name in sorted(os.listdir(self.root))
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.id_list)

    def read_condition(self, image_name):
        fallback_path = os.path.join(self.condition_path, "0000016.png")
        return read_image_512(fallback_path)

    def read_video_frame(self, video_path, frame_idx):
        if not os.path.isfile(video_path):
            print(f"Warning: Step1 video not found, fallback to fixed condition image: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: failed to open Step1 video, fallback to fixed condition image: {video_path}")
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: failed to read frame {frame_idx} from Step1 video, fallback to fixed condition image: {video_path}")
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            return frame
        finally:
            cap.release()

    def read_color_reference(self, image_name):
        if self.initial_image_path is None:
            return None

        video_stem = os.path.splitext(image_name)[0]
        video_path = os.path.join(self.initial_image_path, video_stem, "condition32.mp4")
        return self.read_video_frame(video_path, self.condition_frame_idx)

    def __getitem__(self, idx):
        image_name = self.id_list[idx]
        appearance = read_image_512(os.path.join(self.root, image_name))
        condition = self.read_condition(image_name)
        color_reference = self.read_color_reference(image_name)

        prompt = ""
        if self.transform is not None:
            condition_image = self.transform(appearance)
            condition = self.transform(condition)
            if color_reference is not None:
                color_reference = self.transform(color_reference)
        else:
            condition_image = appearance

        result = {
            "image": condition_image,
            "condition_image": condition_image,
            "condition": condition,
            "text_bg": prompt,
            "text_blip": prompt,
            "image_name": image_name,
            "fea_condition": condition_image,
        }
        if color_reference is not None:
            result["extra_appearance"] = color_reference
        return result


class full_head_clean_inference_final_face(Dataset):
    def __init__(self, condition_path, image_transform=None, inference_image_dataset=None, initial_image_path=None):
        self.condition_path = condition_path
        self.transform = image_transform
        self.inference_image_dataset = inference_image_dataset
        self.initial_image_path = initial_image_path

        back_head_dir = os.path.join(self.inference_image_dataset, "Back_Head")
        self.id_list = [
            file_name
            for file_name in sorted(os.listdir(back_head_dir))
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.frame_rate = len([file_name for file_name in os.listdir(self.condition_path) if file_name.lower().endswith(".png")])

    def __len__(self):
        return len(self.id_list)

    def load_feature_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Error opening feature video file: {video_path}")
            return frames

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(image) if self.transform is not None else image)
            frame_number += 1
        cap.release()
        print(f"Frames extracted: {frame_number}")
        return frames

    def __getitem__(self, idx):
        image_name = self.id_list[idx]
        print("using_id:", image_name)

        appearance_path = os.path.join(self.inference_image_dataset, "input_image", image_name)
        appearance = read_image_512(appearance_path)
        condition_image = self.transform(appearance) if self.transform is not None else appearance

        if self.initial_image_path is not None:
            video_path = os.path.join(
                self.initial_image_path,
                os.path.splitext(image_name)[0],
                f"condition{self.frame_rate}.mp4",
            )
            fea_condition_frames = self.load_feature_video(video_path)
            fea_condition = torch.stack(fea_condition_frames) if len(fea_condition_frames) > 0 else []
        else:
            print("Not using feature Condition")
            fea_condition = []

        drive_img_list = []
        for frame_idx in range(self.frame_rate):
            condition_path = os.path.join(self.condition_path, f"{frame_idx:07d}.png")
            condition = read_image_512(condition_path)
            if self.transform is not None:
                condition = self.transform(condition)
            drive_img_list.append(condition)

        prompt = ""
        result = {
            "image": drive_img_list,
            "condition_image": condition_image,
            "condition": drive_img_list,
            "text_bg": prompt,
            "text_blip": prompt,
            "fea_condition": fea_condition,
            "image_name": image_name,
        }

        back_head_path = os.path.join(self.inference_image_dataset, "Back_Head", image_name)
        assert os.path.isfile(back_head_path), f"File not found: {back_head_path}"
        back_head_image = read_image_512(back_head_path)
        if self.transform is not None:
            back_head_image = self.transform(back_head_image)
        result["extra_appearance"] = back_head_image
        return result
