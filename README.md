![Signer Background](https://github.com/user-attachments/assets/e706e4f4-4b89-4d53-8451-de83a1230fca)
# Signer ✋🤟

Signer is a model that translates signs from video into full sentences through this process:
<br><br>
Input Video --> Boundary Detection --> Landmark Estimation --> Feature Extraction --> Base Model --> Glosses Predictions --> LLM+PrompEngineer --> Output Text 

**Signer** is an Attention-Based Multi-Fusion deep learning module that translates **sign language** into **text**, helping bridge the communication gap between the Deaf/Hard-of-Hearing community and the hearing world. It combines **computer vision** and **Large Language Model** with advanced deep learning techniques to process videos, extract meaningful patterns from gestures, and produce text translations.

---

## 🚀 Features

* 🎥 **Gloss Recognition**: Predicts the full sequence of sign actions (glosses) from the video, providing a complete representation of each gesture.

* 🖐️ **Feature Extraction**: Utilizes [MediaPipe](https://developers.google.com/mediapipe) to accurately detect, track, and extract comprehensive hand, pose, and facial keypoint features from each frame, providing high-fidelity inputs for downstream recognition and translation models.

* ⚡ **Attention-Based Fusion**: Efficiently integrates multi-modal data streams, such as hand, pose, and facial features, using attention mechanisms to improve recognition and translation accuracy.

* 🤖 **Transformer Integration**: Utilizes state-of-the-art transformer architectures to model temporal dependencies within input sequences, enhancing the model's ability to capture contextual information, leverage attention mechanisms, and recognize nuanced patterns in sign gestures.

* 🌍 **Translation to Text**: Converts recognized sign sequences into coherent and contextually meaningful text outputs, completing the end-to-end translation pipeline.

## 🔄 Pipelines

* **VisualizationPipeline:** Primarily designed to visually demonstrate each processing step applied to the input video, while also allowing direct extraction of features for further analysis and model input.

- **PreprocessingPipeline:** Systematically processes selected glosses from the videos folder, converts them into tensors, and stores each in its corresponding gloss folder, serving as the foundational step for subsequent processing in the TrainingPipeline.

- **TrainingPipeline:** Utilizes video frames and landmark stored tensors to systematically extract meaningful features and train the model, providing an end-to-end learning workflow for  sign-to-gloss recognition, The results are subsequently processed on end devices (e.g., mobile applications), The results are further processed on the end device, leveraging large language models to translate the recognized glosses into coherent and meaningful sentence.

- **InferencePipeline:** Performs end-to-end gloss recognition by taking a video path as input and printing the top five predicted glosses along with their associated probabilities, and returning the gloss with high probability

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Signer.git
cd Signer

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies recommended python= 3.12.11
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
Signer/
│── Data sample/ # simple data for testing
│   │──sample.jpeg
│   │──sample.mp4
│── InferencePipeline/
│   ...
│── MediapipeVisualizationPipeline/
│   ...
│── PreprocessinPipeline/
│   ...
│── TrainingPipeline/
│   ...
│── source/
│   │── Inference Notebook.ipynb  # original inference notebook we've used
│   │── Training Notebook.ipynb   # original training notebook we've used
│── inferencePiplineExample.py    # example of how to use inference pipeline
│── visualizationPipelineExample.py # example of the usage of the visaulization pipeline
│── preprocessor.py # example of how to the use of the preprocessing functions
│── requirements.txt              # Python dependencies recommended python 3.12.11
```

---

## ▶️ Usage
> see the examples files [visualizaion](https://github.com/MagdiWaleed/Signer/blob/main/visualizationPipelineExample.py), [inference](https://github.com/MagdiWaleed/Signer/blob/main/inferencePiplineExample.py), and [preprocessor](https://github.com/MagdiWaleed/Signer/blob/main/preprocessor.py), Or you can directly run the notebook without setup using [kaggle](https://www.kaggle.com/) by importing the original notebooks and run them, [Inference Notebook](https://github.com/MagdiWaleed/Signer/blob/main/source/Inference%20Notebook.ipynb) and [Training Notebook](https://github.com/MagdiWaleed/Signer/blob/main/source/Training%20Notebook.ipynb), Kaggle will handel loading the data and the saved models weights.

---

## [AUTSL Dataset](https://cvml.ankara.edu.tr/datasets/)
Ankara University Turkish Sign Language Dataset (AUTSL) is a large-scale, multimode dataset that contains isolated Turkish sign videos. It contains 226 signs that are performed by 43 different signers. There are 38,336 video samples in total. The samples are recorded using Microsoft Kinect v2 in RGB, depth and skeleton formats. We apply some clipping and resizing operations to RGB and depth data and provide them with the resolution of 512×512. The skeleton data contains spatial coordinates, i.e. (x, y), of the 25 junction points on the signer body that are aligned with 512×512 data.

We have only used the RGB format and choosing 26 Glosses from them, We have reached 91 Top-1 accuracy in these glosses, that took about 2 weeks to train in kaggle on only these glosses
Glosses are:
-    0: 'accident'
-    1: 'always'
-    2: 'apologize'
-    3: 'bed'
-    4: 'belt'
-    5: 'breakfast'
-    6: 'bring'
-    7: 'forbidden'
-    8: 'friend'
-    9: 'full'
-    10: 'get_well'
-    11: 'glove'
-    12: 'good'
-    13: 'goodbye'
-    14: 'hurry'
-    15: 'police'
-    16: 'same'
-    17: 'sibling'
-    18: 'single'
-    19: 'thanks'
-    20: 'time'
-    21: 'tomorrow'
-    22: 'wait'
-    23: 'where'
-    24: 'who'
-    25: 'why'
---

## 🤝 Contributing

We welcome contributions from the community! Whether you want to fix bugs, improve documentation, add new datasets, or extend the translation models, your support is appreciated. Please fork the repo, create a feature branch, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License, allowing free use, modification, and distribution with attribution.

---

## 👤 Author

* **Magdi Waleed** – [GitHub](https://github.com/MagdiWaleed) | [LinkedIn](https://www.linkedin.com/in/magdi-waleed) | [Gmail](m.w.m.khalafallah@gmail.com)
* **For More Details**: [documentation.pdf](https://github.com/user-attachments/files/22059116/documentation.pdf)
