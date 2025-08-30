from InferencePipeline.Inference import InferencePipeline
from PIL import Image
import os

inferencePipeline = InferencePipeline(
    {0: 'accident',
    1: 'always',
    2: 'apologize',
    3: 'bed',
    4: 'belt',
    5: 'breakfast',
    6: 'bring',
    7: 'forbidden',
    8: 'friend',
    9: 'full',
    10: 'get_well',
    11: 'glove',
    12: 'good',
    13: 'goodbye',
    14: 'hurry',
    15: 'police',
    16: 'same',
    17: 'sibling',
    18: 'single',
    19: 'thanks',
    20: 'time',
    21: 'tomorrow',
    22: 'wait',
    23: 'where',
    24: 'who',
    25: 'why'},
    )
image =Image.open(os.path.join("Data sample","sample.jpeg"))
result = inferencePipeline.checkLandmarks(image)
print(result)
prediction = inferencePipeline.predict(os.path.join("Data sample","sample.mp4"))
print(prediction)