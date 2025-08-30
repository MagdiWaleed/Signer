
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import functional as F


class Classifier():
    def __init__(self,model,boundaryModel, prediciton_dictanory,model_weights_path=None,boundary_model_weights_path=None,num_partitions= 5,debuging=False,device="cpu"):
       self.prediciton_dictanory = prediciton_dictanory
       self.model = model.to(device)
       self.boundaryDetector = boundaryModel.to(device)
       self.device = device
       if model_weights_path != None:
           if model_weights_path.endswith("checkpoint.pth"):
              if device ==torch.device('cpu'):
                  checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))
                  self.model.load_state_dict(checkpoint["best_model_state_dict"])
              else:
                  checkpoint = torch.load(model_weights_path)
                  self.model.load_state_dict(checkpoint["best_model_state_dict"])
           else:
              self.model.load_state_dict(torch.load(model_weights_path))
       if boundary_model_weights_path != None:
           if device ==torch.device('cpu'):
              self.boundaryDetector.load_state_dict(torch.load(boundary_model_weights_path,map_location=torch.device('cpu')))
           else:
              self.boundaryDetector.load_state_dict(torch.load(boundary_model_weights_path))

       self.num_partitions= num_partitions
       self.debuging = debuging

    def detectBoundary(self,poseLandmarks, max_seq=120*6):
        train = list(poseLandmarks)
        numberofFrames = len(poseLandmarks)
        if numberofFrames < max_seq:
          remaining_frames = max_seq - numberofFrames
          train.extend(torch.zeros((remaining_frames,33,2)))
        train = torch.stack(train)
        train = train.to(self.device)
        output = self.boundaryDetector(train.unsqueeze(0))
        output = torch.argmax(output.squeeze(0), dim= -1)
        print(output)
        boundaries = [ii for ii,tt in enumerate(output) if tt==1]
        return boundaries
    ####
    def predict(self,cropped_images, mediapipefeaturess):
            self.model.eval()
        # if len(cropped_images.shape) == 5:
            seq_predictions = []
            for croped_image,mediapipefeatures in tqdm(zip(cropped_images,mediapipefeaturess)):
                # croped_image = torch.tensor(cropped_image)
                # mediapipefeatures = torch.tensor(mediapipefeatures)

                x_croped_image = []
                x_mediapipefeatures = []

                startingIndcies = torch.linspace(0,int(croped_image.shape[0]/2)-1,steps=self.num_partitions).long()
                limit = len(cropped_images)-1

                for start in startingIndcies:
                    indcies = torch.linspace(start,int(croped_image.shape[0])-1,steps=32).long()
                    indcies  = [i for i in indcies if i <= limit]
                    x_croped_image.append(croped_image[indcies])
                    x_mediapipefeatures.append(mediapipefeatures[indcies])

                x_croped_image = torch.stack(x_croped_image).to(self.device)
                x_mediapipefeatures = torch.stack(x_mediapipefeatures).to(self.device)
                if self.debuging:
                    for i in x_croped_image[0].cpu().permute(0,2,3,1):
                        plt.imshow(i/255)
                        plt.show()
                    print("this is the end of the video")
                # print("cropped images shape: ", x_croped_image.shape)
                # print("medipaipe features shape: ", x_mediapipefeatures.shape)
                with torch.no_grad():
                    mediapipefeatures = mediapipefeatures.float().to(self.device).unsqueeze(0)
                    croped_image = croped_image.float().to(self.device).unsqueeze(0)
                    probabilities = self.model(x_mediapipefeatures,x_croped_image)
                    softmax_probs = F.softmax(probabilities, dim=1)
                    top5_probs, top5_indices = torch.topk(softmax_probs, k=5, dim=1)

                    # Print top-5 for each prediction in the batch
                    for batch_idx in range(top5_indices.size(0)):
                        print(f"\nTop-5 predictions for sample {batch_idx}:")
                        for rank in range(5):
                            pred_idx = top5_indices[batch_idx][rank].item()
                            pred_word = self.prediciton_dictanory[pred_idx]
                            prob = top5_probs[batch_idx][rank].item()
                            print(f"  {rank+1}) {pred_word} - {prob*100:.2f}%")

                    # Still get argmax for majority voting
                    encdoed_output = torch.argmax(probabilities, dim=1)
                    predictions = [self.prediciton_dictanory[prediction.item()] for prediction in encdoed_output]

                    dataDictanory = {word:0 for word in set(predictions)}
                    for i in predictions:
                      dataDictanory[i]+=1
                    frequentLabel =predictions[0]
                    frequency = 0
                    for l,v in dataDictanory.items():
                      if v>frequency:
                        frequentLabel = l
                        frequency = v
                    seq_predictions.append(frequentLabel)
            return seq_predictions

        # else:
        #         x_croped_image = []
        #         x_mediapipefeatures = []


        #         startingIndcies = torch.linspace(0,int(croped_image.size(0)/2)-1,steps=self.num_partitions).long()

        #         for start in startingIndcies:
        #             indcies = torch.linspace(start,int(croped_image.size(0))-1,steps=32).long()
        #             x_croped_image.append(croped_image[indcies])
        #             x_mediapipefeatures.append(mediapipefeatures[indcies])

        #         x_croped_image = torch.stack(x_croped_image).to(device)
        #         x_mediapipefeatures = torch.stack(x_mediapipefeatures).to(device)
        #         # print("cropped images shape: ", x_croped_image.shape)
        #         # print("medipaipe features shape: ", x_mediapipefeatures.shape)
        #         with torch.no_grad():
        #             mediapipefeatures = mediapipefeatures.float().to(device).unsqueeze(0)
        #             croped_image = croped_image.float().to(device).unsqueeze(0)
        #             probabilities = self.model(x_mediapipefeatures,x_croped_image)
        #             encdoed_output = torch.argmax(probabilities,dim=1)
        #             predictions =[self.prediciton_dictanory[prediction.item()] for prediction in encdoed_output]
        #             dataDictanory = {word:0 for word in set(predictions)}
        #             for i in predictions:
        #               dataDictanory[i]+=1
        #             frequentLabel =predictions[0]
        #             frequency = 0
        #             for l,v in dataDictanory.items():
        #               if v>frequency:
        #                 frequentLabel = l
        #                 frequency = v

        #         return frequentLabel, predictions
# processVideo= ProcessVideo(True)
# frames = processVideo.extractFrames("/kaggle/input/new-video-for-testing/WIN_20250609_13_14_53_Pro.mp4")

# _=processVideo.extractMediapipeFeatures(frames)
# start = time.time()
# _,_,_ = processVideo.extractor.extractLandmarks(frames)
# time.time()-start
# Image.fromarray(np.array(frames,dtype= np.uint8))