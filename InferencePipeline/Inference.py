import numpy as np
import mediapipe as mp
from PIL import Image
from .featureExtractor import (
    Extractor,
    FeaturesExtractorFromImage,
    ImageCropper,
    MediapipeFeaturesExtractor,
    ProcessVideo
)
from .Classifier import Classifier
from .VUManager import UploadManager

from .Model import Model
from .BoundarModel import BoundaryDetector

class InferencePipeline():
    def __init__(self,prediciton_dictanory,model_weights_path=None,boundary_model_weights_path=None, target_size = (512,512)):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        mp_face_mesh = mp.solutions.face_mesh
        extractor = Extractor(
            mp_pose,
            mp_hands,
            mp_face_mesh,
            mp_drawing,
            mp_drawing_styles,
            debuging=False
        )
        imageCropper = ImageCropper(False,target_size)
        featureExtractorFromImage = FeaturesExtractorFromImage(
            debuging=False
        )
        mediapipeFeaturesExtractor = MediapipeFeaturesExtractor(
            imageCropper,
            featureExtractorFromImage,
            False,
            target_size
        )
        processVideo = ProcessVideo(
            extractor,
            mediapipeFeaturesExtractor,
            False,
            False,
            target_size
        )
        model = Model()
        boundaryModel = BoundaryDetector()

        classifier = Classifier(
            model,
            boundaryModel,
            prediciton_dictanory,
            model_weights_path,
            boundary_model_weights_path,
            debuging=False
        )
        self.uploadManager = UploadManager(
            processVideo,
            classifier,
        )
    def checkLandmarks(self,image:Image):
        resutl = self.uploadManager.checkLandmarks(image)
        return resutl
    def predict(self,video_path, video=True):
        frames =self.uploadManager.extractFrames(video_path,video=video)
        print("#frames: ",len(frames))
        # for frame in frames:
        #   plt.imshow(frame)
        #   plt.show()
        prediction = self.uploadManager.predict(frames)
        return prediction
