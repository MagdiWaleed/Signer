from torch.utils.data import Dataset
from .MediapipeFeaturesExtractor import MediapipeFeaturesExtractor
import torch 
import numpy as np
from PIL import Image
import random
from torchvision import transforms
import torch.nn.functional as F


class VideoDataset(Dataset):
  def __init__(self,videos_pathes,labels ,agumentation=False,temp_agumentation=False,RGB_only = False,getCropedImageToo= False , cropedtargetsize=(512, 512)):
    
    videos,labels = videos_pathes, labels
   
    if RGB_only:
        new_videos = []
        new_labels = []
        for index in range(len(videos)):
            # print(videos[index],videos[index].split('/')[-1].split('_')[-1])
            if videos[index].split('/')[-1].split('_')[-1].strip() == 'color.pt':
                new_videos.append(videos[index])
                new_labels.append(labels[index])
        videos = new_videos
        labels = new_labels
   
      
    self.videos = videos
    labels = labels

    unique_labels = list(set(labels))
    unique_labels.sort()
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    self.label_to_index = label_to_index
    labels = torch.tensor([label_to_index[label] for label in labels])
    one_hot_encoded_labels = F.one_hot(labels, num_classes=len(unique_labels))
    self.labels = one_hot_encoded_labels
    self.agumentation = agumentation
    self.temp_agumentation = temp_agumentation
    self.cropedtargetsize = cropedtargetsize
    self.getCropedImageToo =getCropedImageToo

    self.mediapipeFeaturesExtractor = MediapipeFeaturesExtractor(debuging=False,target_size=(512,512))

  def __len__(self):
    return len(self.videos)
  
    
  def __getitem__(self,index):    
      landmarks = torch.load(self.videos[index][0],weights_only=False)
      video = torch.load(self.videos[index][1],weights_only=False)
      segmented_video = torch.load(self.videos[index][2],weights_only=False)

      landmarks = np.array(landmarks)
      segmented_video = np.array(segmented_video)
      video = np.array(video)
      
      if self.agumentation:
          randomStarting = torch.randint(0, int(len(video)/2), (1,)).item()
      else:
          randomStarting = 0
      
      indices = torch.linspace(randomStarting, len(video)-1, steps=32).long()
      
      segmented_video = segmented_video[indices]
      landmarks = landmarks[indices]
      video = video[indices]
      if self.agumentation:
          video, segmented_video, agumented_landmarks = self.apply_consistent_augmentation(video,segmented_video, landmarks[:,:-2,:])
          landmarks[:,:-2,:]= agumented_landmarks
      cropped_images, mediapipeFeatures = self.mediapipeFeaturesExtractor.extractFeaturesFromVideo(video, landmarks, segmented_video)
      
      return mediapipeFeatures, cropped_images, self.labels[index]
      

  def apply_consistent_augmentation(self,frames,segmented_frames, landmarks):
    # Generate consistent augmentation parameters for all frames
    flip = random.random() > 0.5
    angle = random.uniform(-0.1, 0.1)  # Degrees
    brightness = random.uniform(0.50, 1.20)
    contrast = random.uniform(0.8, 1.2)
    zoom_factor = random.uniform(1, 1.2)

    augmented_frames = []
    augmented_landmarks = []
    agumented_segmented_frames =[]

    
    for frame, segmented_frame, frame_landmarks in zip(frames,segmented_frames, landmarks):
        
        # Convert frame to PIL Image
        frame_pil = Image.fromarray(frame)
        segmented_frame_pil = Image.fromarray(segmented_frame)
        width, height = frame_pil.size  # Original size

        # Ensure landmarks are a float32 tensor
        if not isinstance(frame_landmarks, torch.Tensor):
            frame_landmarks = torch.tensor(frame_landmarks, dtype=torch.float32)
        else:
            frame_landmarks = frame_landmarks.to(dtype=torch.float32)

        # # --- Step 1: Horizontal Flip ---
        # if flip:
        #     frame_pil = frame_pil.transpose(Image.FLIP_LEFT_RIGHT)
        #     frame_landmarks[:, 0] = width - frame_landmarks[:, 0]  # Flip x-coordinates
        #     segmented_frame_pil = segmented_frame_pil.transpose(Image.FLIP_LEFT_RIGHT)

        # --- Step 2: Rotation ---
        if angle != 0:
            # Rotate image (expand=False to maintain original size)
            frame_pil = frame_pil.rotate(angle, resample=Image.BILINEAR, expand=False)
            segmented_frame_pil = segmented_frame_pil.rotate(angle, resample=Image.BILINEAR, expand=False)
            # Rotate landmarks around the image center
            center = torch.tensor([width / 2, height / 2], dtype=torch.float32)
            angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32))
            cos_theta = torch.cos(angle_rad)
            sin_theta = torch.sin(angle_rad)

            # Rotation matrix
            rotation_matrix = torch.tensor([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ], dtype=torch.float32)

            # Apply rotation: (landmarks - center) @ R + center
            frame_landmarks = (frame_landmarks - center) @ rotation_matrix + center

        # --- Step 3: Brightness/Contrast (no landmark change) ---
        frame_pil = transforms.functional.adjust_brightness(frame_pil, brightness)
        frame_pil = transforms.functional.adjust_contrast(frame_pil, contrast)
        
        segmented_frame_pil = transforms.functional.adjust_brightness(segmented_frame_pil, brightness)
        segmented_frame_pil = transforms.functional.adjust_contrast(segmented_frame_pil, contrast)

        # --- Step 4: Zoom & Crop ---
        # Calculate new dimensions and offsets (as floats)
        new_width = width * zoom_factor
        new_height = height * zoom_factor
        left = (new_width - width) / 2  # Float precision
        top = (new_height - height) / 2

        # Resize image (using integer dimensions)
        new_width_int = int(round(new_width))
        new_height_int = int(round(new_height))
        frame_pil = frame_pil.resize((new_width_int, new_height_int), Image.BILINEAR)

        # Crop to original size (using integer offsets)
        left_int = int(round(left))
        top_int = int(round(top))
        frame_pil = frame_pil.crop((left_int, top_int, left_int + width, top_int + height))
        segmented_frame_pil = segmented_frame_pil.crop((left_int, top_int, left_int + width, top_int + height))

        # Adjust landmarks (using precise float values)
        frame_landmarks = (frame_landmarks * zoom_factor) - torch.tensor([left, top], dtype=torch.float32)

        # --- Finalize ---
        agumented_segmented_frames.append(segmented_frame_pil)
        augmented_frames.append(frame_pil)
        augmented_landmarks.append(frame_landmarks)
    
    return np.array(augmented_frames).astype(dtype=np.uint16),np.array(agumented_segmented_frames).astype(dtype=np.uint16), np.array(augmented_landmarks)

  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")