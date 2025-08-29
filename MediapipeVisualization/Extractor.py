from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

class Extractor():
  def __init__(self,mp_pose,mp_hands,mp_face_mesh,mp_drawing,    mp_drawing_styles,debuging = False):
    self.mp_pose = mp_pose
    self.mp_hands = mp_hands
    self.mp_face_mesh = mp_face_mesh
    self.mp_drawing = mp_drawing
    self.mp_drawing_styles = mp_drawing_styles

    self.BG_COLOR = (0, 0, 0)
    self.debuging = debuging
    self.pose = self.mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=True,
        min_detection_confidence=0.1
        )

    self.hands = self.mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.1
        )

    self.face_mesh = self.mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.1
       )


  def extractLandmarks(self, video):
    print(video.shape)
    if len(video.shape)==4:
        landmarks= []
        segmented_images = []
        images = []
        print(video.shape)
        for image in tqdm(video):
          info = {}
          pose_landmarks = self.extractPose(image)
          hands_landmarks = self.extractHands(image)
          face_landmarks = self.extractFace(image)

          if not pose_landmarks["pose_landmark"]:
            info["pose_landmarks"] = None
            info["body_segmentation"] = pose_landmarks["body_segmentation"]
          else:
            info["pose_landmarks"] = pose_landmarks["pose_landmark"]
            info["body_segmentation"] = pose_landmarks["body_segmentation"]

          if not hands_landmarks:
            info["hands_landmarks"] = {
                "right_hand":None,
                "left_hand":None,
                "landmarks":None
            }
          else:
            info["hands_landmarks"] = hands_landmarks

          if not face_landmarks:
            info["face_landmarks"] = None
          else:
            info["face_landmarks"] = face_landmarks

          checker = self.check(info)
          x_y_landmarks = self.get_x_y_landmarks_scaled_up(info,image)

          right_hand = 0
          left_hand = 0
          face = 0
          pose = 0
          if checker["right_hand_landmarks"]:
            right_hand = 1
          if checker["left_hand_landmarks"]:
            left_hand = 1
          if checker["pose_landmarks"]:
              pose = 1
          if checker["face_landmarks"]:
              face =1
          x_y_landmarks.append([right_hand,pose])
          x_y_landmarks.append([face,left_hand])
          landmarks.append(x_y_landmarks)
          segmented_image = self.getSegmentedImage(info,image)
          segmented_images.append(segmented_image)
          images.append(image)
          if self.debuging:
              print(checker)
              self.showLandmarks(info,image)
              plt.imshow(segmented_image)
              plt.show()
        return landmarks, segmented_images, images
    else:
      info = {}
      pose_landmarks = self.extractPose(image)
      hands_landmarks = self.extractHands(image)
      face_landmarks = self.extractFace(image)

      if not pose_landmarks["pose_landmark"]:
        info["pose_landmarks"] = None
        info["body_segmentation"] = pose_landmarks["body_segmentation"]
      else:
        info["pose_landmarks"] = pose_landmarks["pose_landmark"]
        info["body_segmentation"] = pose_landmarks["body_segmentation"]

      if not hands_landmarks:
        info["hands_landmarks"] = {
            "right_hand":None,
            "left_hand":None,
            "landmarks":None
        }
      else:
        info["hands_landmarks"] = hands_landmarks

      if not face_landmarks:
        info["face_landmarks"] = None
      else:
        info["face_landmarks"] = face_landmarks

      checker = self.check(info)
      x_y_landmarks = self.get_x_y_landmarks_scaled_up(info,image)

      right_hand = 0
      left_hand = 0
      face = 0
      pose = 0
      if checker["right_hand_landmarks"]:
        right_hand = 1
      if checker["left_hand_landmarks"]:
        left_hand = 1
      if checker["pose_landmarks"]:
          pose = 1
      if checker["face_landmarks"]:
          face =1
      x_y_landmarks.append([right_hand,pose])
      x_y_landmarks.append([face,left_hand])
      if self.debuging:
          print(checker)
          self.showLandmarks(info,image)
    return x_y_landmarks, self.getSegmentedImage(info,image), image


  def get_x_y_landmarks_scaled_up(self,info,image):
    landmarks = self.get_x_y_landmarks(info,image)
    shape = image.shape
    landmarks = [[t[0]*shape[1],t[1]*shape[0]] for t in landmarks]
    return landmarks
  def get_x_y_landmarks(self,info, image):

    x_y_landmarks = []
    if info["hands_landmarks"]["right_hand"] != None:
      for landmark in info["hands_landmarks"]["right_hand"].landmark:
        x_y_landmarks.append([landmark.x,landmark.y])
    else:
      x_y_landmarks.extend(list(np.zeros((21,2))))


    if info["pose_landmarks"] != None:
      for landmark in info["pose_landmarks"].landmark:
        x_y_landmarks.append([landmark.x,landmark.y])
    else:
      x_y_landmarks.extend(list(np.zeros((33,2))))


    if info["face_landmarks"] != None:
      for landmark in info["face_landmarks"].landmark:
        x_y_landmarks.append([landmark.x,landmark.y])
    else:
      x_y_landmarks.extend(list(np.zeros((478,2))))


    if info["hands_landmarks"]["left_hand"] != None:
      for landmark in info["hands_landmarks"]["left_hand"].landmark:
        x_y_landmarks.append([landmark.x,landmark.y])
    else:
      x_y_landmarks.extend(list(np.zeros((21,2))))

    return x_y_landmarks

  def check(self,info):
    return {
        "pose_landmarks":info["pose_landmarks"] !=None,
        "right_hand_landmarks":info["hands_landmarks"]["right_hand"] !=None,
        "left_hand_landmarks":info["hands_landmarks"]["left_hand"] !=None,
        "face_landmarks":info["face_landmarks"]!=None
    }
  def extractPose(self,image):
      results = self.pose.process(image)
      if not results.pose_landmarks:
        return {
            "pose_landmark":None,
            "body_segmentation": results.segmentation_mask
            }
      return {
            "pose_landmark":results.pose_landmarks,
            "body_segmentation": results.segmentation_mask
            }


  def extractHands(self,image):
      results = self.hands.process(image)

      if not results.multi_hand_landmarks:
        return None
      data = {
          "right_hand":None,
          "left_hand":None,
          "landmarks":results.multi_hand_landmarks
          }
      for handedness, landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
          label = handedness.classification[0].label
          if label == "Right":
              data["right_hand"] = landmarks
          elif label == "Left":
              data["left_hand"] = landmarks
      return data


  def extractFace(self,image):
      results = self.face_mesh.process(image)

      if not results.multi_face_landmarks:
        return None
      return results.multi_face_landmarks[0]

  def showSegmentedBody(self):
    if self.debuging:
      condition = np.stack((self.info["body_segmentation"],) * 3, axis=-1) > 0.1
      bg_image = np.zeros(self.image.shape, dtype=np.uint8)
      bg_image[:] = self.BG_COLOR
      annotated_image = np.where(condition, self.image, bg_image)
      plt.imshow(annotated_image)
      plt.show()
    else:
      print("enable debuging")

  def getSegmentedImage(self,info,image):
      if  info['body_segmentation'] is not None:
          condition = np.stack((info["body_segmentation"],) * 3, axis=-1) > 0.1
          bg_image = np.zeros(image.shape, dtype=np.uint8)
          bg_image[:] = self.BG_COLOR
          annotated_image = np.where(condition, image, bg_image)
          return annotated_image
      else:
          print("there is no segmented image for this frame")
          return list(np.zeros((512,512,3)))

  def showLandmarks(self,info,image):
    image_with_landmarks = image.copy()
    if info["pose_landmarks"] != None:

      self.mp_drawing.draw_landmarks(
              image_with_landmarks,
              info["pose_landmarks"],
              self.mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
    if info["hands_landmarks"]['landmarks'] != None:
      for hand_landmarks in info["hands_landmarks"]['landmarks']:
        self.mp_drawing.draw_landmarks(
            image_with_landmarks,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style())

    if info["face_landmarks"] != None:
        self.mp_drawing.draw_landmarks(
            image=image_with_landmarks,
            landmark_list=info["face_landmarks"],
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        self.mp_drawing.draw_landmarks(
            image=image_with_landmarks,
            landmark_list=info["face_landmarks"],
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_contours_style())
        self.mp_drawing.draw_landmarks(
            image=image_with_landmarks,
            landmark_list=info["face_landmarks"],
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    plt.imshow(image_with_landmarks)
    plt.show()