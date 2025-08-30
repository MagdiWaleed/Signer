import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image

class MediapipeFeaturesExtractor():
  def __init__(self,imageCropper,featuresExtractorFromImage, debuging =False):
    self.imageCropper = imageCropper
    self.featuresExtractorFromImage = featuresExtractorFromImage
    self.debuging = debuging


  def extractFeaturesFromVideo(self, frames, landmarks, segmented_frames):
    cropped_images = []
    mediapipeFeatures = []
    previous_scaled_up_right_hand_landmarks = [0]
    previous_scaled_up_pose_landmarks = [0]
    previous_scaled_up_face_landmarks = [0]
    previous_scaled_up_left_hand_landmarks = [0]

    # for frame in tqdm(frames):
    #   self.extractor.extractLandmarks(frame)
    #   checkers.append(self.extractor.check())
    #   fframes.append(self.extractor.image)
    #   landmarks.append(self.extractor.get_x_y_landmarks_scaled_up())
    for frame,landmark,segmented_frame in zip(frames,landmarks, segmented_frames):

      checker = {
          "right_hand_landmarks":(landmark[-2][0]==1),
          "left_hand_landmarks":(landmark[-1][1]==1),
          "pose_landmarks":(landmark[-2][1]==1),
          "face_landmarks":(landmark[-1][0]==1),
      }
      if not checker['right_hand_landmarks'] and np.sum(previous_scaled_up_right_hand_landmarks) != 0 :
        landmark[:21,:] = previous_scaled_up_right_hand_landmarks

      if not checker['left_hand_landmarks'] and np.sum(previous_scaled_up_right_hand_landmarks) != 0 :
        landmark[532:-2,:] = previous_scaled_up_left_hand_landmarks

      if not checker['pose_landmarks'] and np.sum(previous_scaled_up_pose_landmarks) != 0:
        landmark[21:54,:]= previous_scaled_up_pose_landmarks
      if not checker['face_landmarks'] and np.sum(previous_scaled_up_face_landmarks) != 0:
        landmark[54:532,:]= previous_scaled_up_face_landmarks
      
      previous_scaled_up_right_hand_landmarks = landmark[:21,:]
      previous_scaled_up_pose_landmarks = landmark[21:54,:]
      previous_scaled_up_face_landmarks = landmark[54:532,:]
      previous_scaled_up_left_hand_landmarks = landmark[532:-2,:]

      new_frame, adjusted_landmark = self.imageCropper.cropImage(frame, landmark[:-2], checker)
      cropped_image, _ = self.imageCropper.cropImage(segmented_frame, landmark[:-2], checker)
      mediapipeFeature = self.featuresExtractorFromImage.extractFeatures(new_frame.permute(1,2,0).numpy().astype(dtype=np.uint16),checker,adjusted_landmark)


      cropped_images.append(cropped_image)
      mediapipeFeatures.append(torch.tensor(mediapipeFeature))
      if self.debuging:
        self.showImages()
    return torch.stack(cropped_images), torch.stack(mediapipeFeatures)

  def showImages(self):
    if self.debuging:
      self.imageCropper.showImage()
      self.featuresExtractorFromImage.showFeatures()
    else:
      print("sett debuging flag to True")

