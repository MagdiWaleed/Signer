import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ImageCropper():
  def __init__(self,debuging,target_size = (512, 512) ):
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

class FeaturesExtractorFromImage():
  def __init__(self,debuging):
    if debuging:
      self.info = {
          "left_hand_image":None,
          "right_hand_image":None,
          "vertical":None,
          "horizontal":None,
          "left_distance":None,
          "right_distance":None
      }

    self.debuging= debuging
    pass

  def extractFeatures(self,image, landmarks_checker, scaled_landmarks):

    if not landmarks_checker["pose_landmarks"] and not landmarks_checker["right_hand_landmarks"] and not landmarks_checker["left_hand_landmarks"] and not landmarks_checker["face_landmarks"]:
      return list(np.zeros((39514)))
    scaled_landmarks = scaled_landmarks
    # right hands landmarks:  21
    # pose landmarks:  33
    # face landmarks:  478
    # left hands landmarks:  21
    right_hand_landmarks,pose_landmarks,face_landmarks,left_hand_landmarks = scaled_landmarks[:21,:],scaled_landmarks[21:54,:],scaled_landmarks[54:532,:],scaled_landmarks[532:,:]
    # print("this now in mediapipe")
    # for i in range(len(scaled_landmarks[21:54,:])):
    #       print("index: ",i,") pose: ",scaled_landmarks[21:54,:][i])
    
    vertical_regoins, horizontal_regions, right_hands_box_cordinations, pose_box_cordinations, face_box_cordinations, left_hands_box_cordinations =self.handsRegoins(
      landmarks_checker,
      right_hand_landmarks,
      pose_landmarks,
      face_landmarks,
      left_hand_landmarks,
      image
      )

    left_distance,right_distance,left_hand_image, right_hand_image = self.getHandsDistanceFromFace(
      landmarks_checker,
      right_hands_box_cordinations,
      pose_box_cordinations,
      face_box_cordinations,
      left_hands_box_cordinations,
      image
      )

    LR_hand_regions = np.array([vertical_regoins,horizontal_regions])

    # print(left_hand_image.reshape(-1).shape)
    # print(right_hand_image.reshape(-1).shape)
    # print(LR_hand_regions.reshape(-1).shape)
    # print(left_distance.reshape(-1).shape)
    # print(right_distance.reshape(-1).shape)
    # print(right_hand_landmarks.reshape(-1).shape)
    # print(pose_landmarks.reshape(-1).shape)
    # print(face_landmarks.reshape(-1).shape)
    # print(left_hand_landmarks.reshape(-1).shape)

    # print("maximum inside extract functino: ",left_hand_image.max())
    # plt.imshow( left_hand_image)
    # plt.show()


    frame_features = np.concatenate([left_hand_image.reshape(-1),right_hand_image.reshape(-1),LR_hand_regions.reshape(-1),left_distance.reshape(-1),right_distance.reshape(-1), right_hand_landmarks.reshape(-1),pose_landmarks.reshape(-1),face_landmarks.reshape(-1),left_hand_landmarks.reshape(-1)])
    return torch.tensor(frame_features)

  def getHandsDistanceFromFace(self,landmarks_checker,right_hands_box_cordinations, pose_box_cordinations, face_box_cordinations, left_hands_box_cordinations, image):
      margin = 10

      left_hand_image= image[left_hands_box_cordinations[0][1]-margin:left_hands_box_cordinations[1][1]+margin,
                        left_hands_box_cordinations[0][0]-margin:left_hands_box_cordinations[1][0]+margin]

      right_hand_image= image[right_hands_box_cordinations[0][1]-margin:right_hands_box_cordinations[1][1]+margin,
                        right_hands_box_cordinations[0][0]-margin:right_hands_box_cordinations[1][0]+margin]
      left_hand_image = cv2.resize(left_hand_image, (80,80)) if left_hand_image.shape[0] > 0 and left_hand_image.shape[1] > 0 else np.zeros((80,80,3))
      right_hand_image = cv2.resize(right_hand_image, (80,80)) if right_hand_image.shape[0] > 0 and right_hand_image.shape[1] > 0 else np.zeros((80,80,3))
      distance_between_left_face = np.array([np.abs(face_box_cordinations[0][0]-left_hands_box_cordinations[0][0]),np.abs(face_box_cordinations[1][1] -left_hands_box_cordinations[1][1])])
      distance_between_right_face = np.array([np.abs(face_box_cordinations[0][0]-right_hands_box_cordinations[0][0]),np.abs(face_box_cordinations[1][1] -right_hands_box_cordinations[1][1])])
      if self.debuging:
        self.info['left_distance'] = distance_between_left_face
        self.info['right_distance'] = distance_between_right_face
        self.info['left_hand_image'] = left_hand_image
        self.info['right_hand_image'] = right_hand_image
      return distance_between_left_face,distance_between_right_face,(left_hand_image),(right_hand_image)


  def handsRegoins(self, landmarks_checker, right_hand_landmarks,pose_landmarks,face_landmarks,left_hand_landmarks,image):
    right_hands_box_cordinations = np.min(right_hand_landmarks,axis = 0).astype(int), np.max(right_hand_landmarks,axis=0).astype(int)#min bottom left - max top right
    pose_box_cordinations = np.min(pose_landmarks,axis = 0).astype(int), np.max(pose_landmarks,axis=0).astype(int)#min bottom left - max top right
    face_box_cordinations = np.min(face_landmarks,axis = 0).astype(int), np.max(face_landmarks,axis=0).astype(int)#min bottom left - max top right
    left_hands_box_cordinations = np.min(left_hand_landmarks,axis = 0).astype(int), np.max(left_hand_landmarks,axis=0).astype(int)#min bottom left - max top right

    if not landmarks_checker['face_landmarks']:
      new_box_cordinations = np.array([pose_landmarks[8],pose_landmarks[7],pose_landmarks[10],pose_landmarks[9]])
      face_box_cordinations = np.min(new_box_cordinations,axis = 0).astype(int),np.max(new_box_cordinations,axis=0).astype(int)
        
    if not landmarks_checker["right_hand_landmarks"]:
      new_box_cordinations = np.array([pose_landmarks[21],pose_landmarks[19],pose_landmarks[15],pose_landmarks[17]])
      right_hands_box_cordinations = np.min(new_box_cordinations,axis = 0).astype(int),np.max(new_box_cordinations,axis=0).astype(int)
    if not landmarks_checker["left_hand_landmarks"]:
      new_box_cordinations = np.array([pose_landmarks[20],pose_landmarks[22],pose_landmarks[18],pose_landmarks[16]])
      left_hands_box_cordinations = np.min(new_box_cordinations,axis = 0).astype(int),np.max(new_box_cordinations,axis=0).astype(int)

    vertical_hands_locations =[]
    for hand in [left_hands_box_cordinations,right_hands_box_cordinations]:
      if hand[0][1] <=face_box_cordinations[0][1]:
        vertical_hands_locations.append(1)
      elif hand[0][1] >=face_box_cordinations[0][1] and hand[0][1] < pose_box_cordinations[0][1]:
        vertical_hands_locations.append(2)
      elif hand[0][1] <face_box_cordinations[1][1] and hand[0][1] >= pose_box_cordinations[0][1]:
        vertical_hands_locations.append(3)
      elif hand[0][1] >pose_box_cordinations[0][1] and hand[0][1]>= face_box_cordinations[1][1] and hand[0][1]< (pose_box_cordinations[0][1]/2)+pose_box_cordinations[0][1] :
        vertical_hands_locations.append(4)
      else:
        vertical_hands_locations.append(5)
    if self.debuging:
      print("this in mediapipe")
      m = image.copy()
      top_left, bottom_right = left_hands_box_cordinations
      cx = int((top_left[0] + bottom_right[0]) / 2)
      cy = int((top_left[1] + bottom_right[1]) / 2)
      center = (cx, cy)
      cv2.circle(m, center, radius=5, color=(0, 0, 255), thickness=-1)
      cv2.rectangle(m, tuple(face_box_cordinations[0]), tuple(face_box_cordinations[1]), (255, 0, 0), 2)
      cv2.rectangle(m, tuple(left_hands_box_cordinations[0]), tuple(left_hands_box_cordinations[1]), (255, 0, 255), 2)
      cv2.rectangle(m, tuple(right_hands_box_cordinations[0]), tuple(right_hands_box_cordinations[1]), (255, 255, 0), 2)
      cv2.rectangle(m, tuple(pose_box_cordinations[0]), tuple(pose_box_cordinations[1]), (255, 255, 255), 2)
      print("this is tuple: ",tuple(pose_box_cordinations[0]), tuple(pose_box_cordinations[1]))
      # for i in range(len(pose_landmarks)):
      #     print("index: ",i,") pose: ",pose_landmarks[i])
      plt.imshow(m)
      plt.show()


    horizontal_hands_locations = []
    for hand in [left_hands_box_cordinations,right_hands_box_cordinations]:
      if hand[0][0]< pose_box_cordinations[0][0]:
        horizontal_hands_locations.append(1)
      elif hand[0][0] >= pose_box_cordinations[0][0] and hand[0][0] < face_box_cordinations[0][0]:
        horizontal_hands_locations.append(2)
      elif hand[0][0] >= face_box_cordinations[0][0] and hand[0][0] < face_box_cordinations[1][0]:
        horizontal_hands_locations.append(3)
      elif hand[0][0] >=face_box_cordinations[1][0] and hand[0][0]< pose_box_cordinations[1][0]:
        horizontal_hands_locations.append(4)
      else:
        horizontal_hands_locations.append(5)
    if self.debuging:
      self.info['horizontal'] = horizontal_hands_locations
      self.info['vertical'] = vertical_hands_locations
    return vertical_hands_locations,horizontal_hands_locations, right_hands_box_cordinations, pose_box_cordinations, face_box_cordinations, left_hands_box_cordinations

  def showFeatures(self):
    print("left hand distance from the face: ", self.info['left_distance'])
    print("right hand distance from the face: ", self.info['right_distance'])
    print("vertical hands regions: ",self.info["vertical"])
    print("horizontal hands regions: ",self.info["horizontal"])
    print("left hand image: ")
    plt.imshow(self.info["left_hand_image"])
    plt.show()
    print("right hand image: ")
    plt.imshow(self.info["right_hand_image"])
    plt.show()

class MediapipeFeaturesExtractor():
  def __init__(self,debuging =True,target_size = (512,512)):      
    self.imageCropper = ImageCropper(debuging,target_size )
    self.featuresExtractorFromImage = FeaturesExtractorFromImage(debuging)
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
      
      cropped_image, _ = self.imageCropper.cropImage(segmented_frame, landmark[:-2], checker)
      new_frame, adjusted_landmark = self.imageCropper.cropImage(frame, landmark[:-2], checker)
      mediapipeFeature = self.featuresExtractorFromImage.extractFeatures(new_frame.permute(1,2,0).numpy().astype(dtype=np.uint16),checker,adjusted_landmark)
      cropped_images.append(cropped_image)
      mediapipeFeatures.append(mediapipeFeature)
      if self.debuging:
        self.showImages()
    return torch.stack(cropped_images), torch.stack(mediapipeFeatures)

  def showImages(self):
    if self.debuging:
      self.imageCropper.showImage()
      self.featuresExtractorFromImage.showFeatures()
    else:
      print("sett debuging flag to True")
