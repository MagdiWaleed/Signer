from torchvision import transforms
import cv2
import numpy as np
import torch
import os 
from tqdm import tqdm
from PIL import Image

def _preprocess(video_path):
      transform = transforms.Compose(
              [
                  transforms.Resize((512, 512)),
                  transforms.ToTensor(),
              ]
          )

      valid_video=True
      cap = cv2.VideoCapture(video_path)

      fps = cap.get(cv2.CAP_PROP_FPS)
      frames = []
      if fps == 0:
        valid_video=False
        print("End of video or error occurred.")
        return None

      while True:
        ret, frame = cap.read()
        valid_video=True
        if not ret:
            break  # End of video

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

      if valid_video:
        frames = np.array(frames, dtype=np.uint8)

        tensor = torch.tensor(frames)

        pil_images = [Image.fromarray(frame.numpy()) for frame in tensor]

        transformed_images = [transform(image) for image in pil_images]

        transformed_images_tensor = torch.stack(transformed_images)


        return transformed_images_tensor
      else:
        return None
import shutil

def creat_gloss_folder(data_path,output_folder,chooser=[],RGB_Only= False):
    data_dict = {}
    for gloss in tqdm(os.listdir(data_path)):
      data_dict[gloss] = []
    for gloss in tqdm(os.listdir(data_path)):
      for video in os.listdir(os.path.join(data_path,gloss)):
        data_dict[gloss].append(os.path.join(data_path,output_folder,video))

    for gloss in data_dict.keys():
      os.makedirs(os.path.join(output_folder,gloss), exist_ok=True)
    for gloss in gloss.keys():
      print("working on gloss: ",gloss)
      for video_path in data_dict[gloss]:
        base_name =  video_path.split('/')[-1]
        frames = _preprocess(video_path)
        torch.save(frames, os.path.join(output_folder,gloss,base_name+'.pt'))



