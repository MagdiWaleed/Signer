from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import cv2


class UploadManager():
    def __init__(self,processVideo, classifier,  extractOneFromEach= 2, intersectionRatio = 0.5, predict_after = 40,target_size=(512,512)):
        self.target_size = target_size
        self.extractOneFromEach = extractOneFromEach
        self.predict_after = predict_after
        self.intersectionRatio = intersectionRatio
        self.transform = transforms.CenterCrop((512, 512))

        self.mediapipeFeatures =[]
        self.cropedImages = []

        self.processVideo = processVideo
        self.classifier = classifier

    def checkLandmarks(self,frame):
        checker = self.processVideo.extractor.extractLandmarks(np.array(frame))
        return checker
    def extractFrames(self,video_path, video=False):
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
        # plt.imshow(frame)
        # plt.show()
        # Resize the frame to the target size before appending
        if not video:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame = Image.fromarray(frame)
        frame = self.transform(frame)
        # plt.imshow(frame)
        # plt.show()
        frames.append(frame)

      if valid_video:
        frames = np.array(frames, dtype=np.uint8)

        tensor = torch.tensor(frames)

        # transformed_images = [self.top_center_crop_cv2(image) for image in tensor]

        return tensor
      else:
        print("there is error in the frames")
        return None

    ###
    def predict(self, frames):
        landmarks, segmented_frames, frames =self.processVideo.extractLandmarks(np.array(frames))
        print("landmarks extracted successfully ")
        landmarks = np.array(landmarks)
        frames = np.array(frames)
        segmented_frames = np.array(segmented_frames)
        cropped_images, mediapipeFeatures, updated_adjusted_landmarks = self.processVideo.extractFeaturesFromVideo( frames, landmarks, segmented_frames)
        print("mediapipe features extracted successfully ")

        pose_landmarks = updated_adjusted_landmarks[:,21:54,:].clone().float()
        boundaries_dash = self.classifier.detectBoundary(pose_landmarks)
        print("boundaries is: ", boundaries_dash)

        boundaries = []
        for tt in range(len(boundaries_dash)):
          if tt!= 0 and boundaries_dash[tt]== boundaries_dash[tt-1]:
            continue
          else:
            boundaries.append(boundaries_dash[tt])

        def remove_consecutive_increases(lst):
            if not lst:
                return []

            result = [lst[0]]

            for i in range(1, len(lst)):
                if lst[i] != lst[i-1] + 1:
                    result.append(lst[i])

            return result



        boundaries = remove_consecutive_increases(boundaries)

        print(boundaries)

        print(cropped_images.shape)
        cropped_images_compo = []
        mediapipeFeatures_compo = []
        start = 0
        for end in boundaries:
            end = int(end)
            cropped_images_compo.append(cropped_images[start:end, :, :, :])
            mediapipeFeatures_compo.append(mediapipeFeatures[start:end, :])
            start = end
        cropped_images_compo = cropped_images_compo
        mediapipeFeatures_compo = mediapipeFeatures_compo
        print("generated sequences successfully ")

        predictions = self.classifier.predict(cropped_images_compo, mediapipeFeatures_compo)
        print("predictions successfully ")
        return predictions
    
    