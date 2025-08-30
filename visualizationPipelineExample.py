from MediapipeVisualizationPipeline.Visualizer import VisualizationPipeline
import os
if __name__ == "__main__":
    visualizer = VisualizationPipeline()# JustVisualize --> True to visualize and process one frame
    (cropped_images, mediapipeFeatures),( landmarks, segmented_frames, frames) =visualizer.process(os.path.join("Data sample","sample.mp4"))