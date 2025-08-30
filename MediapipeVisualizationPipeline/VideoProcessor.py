from torchvision import transforms
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
class ProcessVideo():
  def __init__(self,extractor, mediapipeFeaturesExtractor,target_size=(512,512)):
    self.extractor = extractor
    self.mediapipeFeaturesExtractor = mediapipeFeaturesExtractor
    self.target_size = target_size

  def extractLandmarks(self, frames):
      landmarks, segmented_image, image = self.extractor.extractLandmarks(frames)
      return landmarks, segmented_image, image

  def extractFeaturesFromVideo(self, frames, landmarks, segmented_frames):
    cropped_images, mediapipeFeatures = self.mediapipeFeaturesExtractor.extractFeaturesFromVideo(frames, landmarks, segmented_frames)
    return cropped_images, mediapipeFeatures
  def extractFrames(self,video_path):
      transform = transforms.Compose(
              [
                  transforms.Resize(self.target_size),
              ]
          )
      frames = []
      valid_video=True
      cap = cv2.VideoCapture(video_path)
      fps = cap.get(cv2.CAP_PROP_FPS)
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

        pil_images = [Image.fromarray(frame) for frame in frames]

        transformed_images = [transform(image) for image in pil_images]

        return transformed_images
      else:
        print("there is error in the frames")
        return None