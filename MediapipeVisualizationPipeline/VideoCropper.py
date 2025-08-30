import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt


class ImageCropper():
  def __init__(self,debuging=False,target_size = (512, 512) ):
    self.target_size=target_size
    self.margin=20
    self.debuging= debuging
    self.transform=None
    self.normalized=False
    if debuging:
      self.cropped_image = None

  def cropImage(self,image, landmarks, checker):
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        image = (image).astype(np.uint8)
    if torch.is_tensor(landmarks):
        landmarks_np = landmarks.cpu().numpy()
    else:
        landmarks_np = np.array(landmarks).copy()

    # Ensure landmarks shape is (N, 2)
    if landmarks_np.ndim == 1 and landmarks_np.shape[0] % 2 == 0:
        landmarks_np = landmarks_np.reshape(-1, 2)
    elif landmarks_np.ndim != 2 or landmarks_np.shape[1] != 2:
        raise ValueError(f"Invalid landmarks shape: {landmarks_np.shape}")

    frame = image
    lm = landmarks_np
    if not checker['face_landmarks']:
        variable = lm[6+21]
        variable[1]-=50
        lm[54:532,:] = variable


    if not checker['right_hand_landmarks']:
        lm[:21,:] = lm[17+21]

    if not checker['left_hand_landmarks']:
        lm[532:,:] = lm[18+21]
    # print("this in cropped image")
    # for i in range(len(lm[21:54,:])):
    #       print("index: ",i,") pose: ",lm[21:54,:][i])

    cropped_image, (x_min, x_max, y_min, y_max) = self.detect_and_crop_person(frame, lm, margin=self.margin)
    if cropped_image.size == 0:
        adjusted_landmarks = np.zeros_like(landmarks_np)
        output_image = torch.zeros(3, self.target_size[1], self.target_size[0])
        return output_image, adjusted_landmarks

    processed_img, scale, (x_offset, y_offset) = self.resize_and_pad(cropped_image, target_size=self.target_size)

    if self.normalized:
        h, w, _ = frame.shape
        x_coords = lm[:, 0] * w
        y_coords = lm[:, 1] * h
    else:
        x_coords = lm[:, 0]
        y_coords = lm[:, 1]

    adjusted_x = (x_coords - x_min) * scale + x_offset
    adjusted_y = (y_coords - y_min) * scale + y_offset

    target_w, target_h = self.target_size
    # out_of_bounds = (
    #     (adjusted_x < 0) | (adjusted_x >= target_w) |
    #     (adjusted_y < 0) | (adjusted_y >= target_h)
    # )
    # adjusted_x[out_of_bounds] = 0.0
    # adjusted_y[out_of_bounds] = 0.0

    if self.normalized:
        adjusted_x /= target_w
        adjusted_y /= target_h

    adjusted_landmarks = np.stack([adjusted_x, adjusted_y], axis=-1)

    if self.transform:
        processed_img = self.transform(processed_img)
    else:
        processed_img = torch.from_numpy(processed_img).permute(2, 0, 1).float()
    if self.debuging:
      self.cropped_image = processed_img
    return processed_img, adjusted_landmarks

  def resize_and_pad(self,image, target_size=(512, 512)):
      target_w, target_h = target_size
      h, w, _ = image.shape
      if h == 0 or w == 0:
          return np.zeros((target_h, target_w, 3), dtype=np.uint8), 1.0, (0, 0)
      padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
      scale = min(target_w / w, target_h / h)
      new_w, new_h = int(w * scale), int(h * scale)
      if new_w == 0 or new_h == 0:
          return padded_image, 1.0, (0, 0)
      resized_image = cv2.resize(image, (new_w, new_h))
      x_offset = (target_w - new_w) // 2
      y_offset = (target_h - new_h) // 2
      padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
      return padded_image, scale, (x_offset, y_offset)

  def detect_and_crop_person(self,frame, lm, margin=10):
      h, w, _ = frame.shape
      if self.normalized:
          x_coords = lm[:, 0] * w
          y_coords = lm[:, 1] * h
      else:
          x_coords = lm[:, 0]
          y_coords = lm[:, 1]
      x_min = int(max(np.min(x_coords) - margin, 0))
      x_max = int(min(np.max(x_coords) + margin, w))
      y_min = int(max(np.min(y_coords) - margin, 0))
      y_max = int(min(np.max(y_coords) + margin, h))
      if x_max <= x_min or y_max <= y_min:
          return frame, (0, w, 0, h)
      crop = frame[y_min:y_max, x_min:x_max]
      if crop.size == 0:
          return frame, (0, w, 0, h)
      return crop, (x_min, x_max, y_min, y_max)
  def showImage(self):
    print("this cropped image: ")
    image = self.cropped_image.permute(1,2,0).numpy().astype(dtype=np.uint16)
    plt.imshow(image)
    plt.show()