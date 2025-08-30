import numpy as np
import mediapipe as mp
from .Extractor import Extractor
from .VideoCropper import ImageCropper
from .FeaturesExtractorFromImage import FeaturesExtractorFromImage
from .MediapipeFeatureExtractor import MediapipeFeaturesExtractor
from .VideoProcessor import ProcessVideo

class VisualizationPipeline():
    def __init__(self,JustVisualize=False,from_index=0,to_index=1,frames_size = (512,512)):
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
            debuging=JustVisualize
        )
        imageCropper = ImageCropper(
            debuging=JustVisualize,
            target_size=frames_size
            )
        featuresExtractorFromImage = FeaturesExtractorFromImage(
            debuging=JustVisualize
        )
        mediapipeFeaturesExtractor = MediapipeFeaturesExtractor(
            imageCropper,
            featuresExtractorFromImage,
            debuging=JustVisualize
            )
      
        self._processVideo = ProcessVideo(
            extractor,
            mediapipeFeaturesExtractor,
            target_size=frames_size
        )
        self.JustVisualize = JustVisualize
        self.from_index= from_index
        self.to_index = to_index
    def process(self,video_path):
        frames =self._processVideo.extractFrames(video_path)
        print("number of frames extracted is : ",len(frames))
        if self.JustVisualize is not None:
            landmarks, segmented_frames, frames = self._processVideo.extractLandmarks(np.array(frames)[self.from_index:self.to_index])
        else:
            landmarks, segmented_frames, frames = self._processVideo.extractLandmarks(np.array(frames))
        landmarks = np.array(landmarks)
        frames = np.array(frames)
        segmented_frames = np.array(segmented_frames)

        cropped_images, mediapipeFeatures =self._processVideo.extractFeaturesFromVideo( frames, landmarks, segmented_frames)
        return (cropped_images, mediapipeFeatures),( landmarks, segmented_frames, frames)
        
        
