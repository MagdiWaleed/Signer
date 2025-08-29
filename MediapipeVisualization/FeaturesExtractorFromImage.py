import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt

class FeaturesExtractorFromImage():
  def __init__(self,debuging =False):
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