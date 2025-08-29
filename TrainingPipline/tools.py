import torch
from tqdm import tqdm
import numpy as np
import time
import json 
import os

def predict(model,test_loader,device):
    predictions = []
    actual = []
    model.eval()
    with torch.no_grad():
        for x,croped_image,y in tqdm(test_loader):    
            x = x.float().to(device) # batch_size, num_frames, channels, height, width
            croped_image = croped_image.float().to(device)
            outputs = model(x,croped_image)
            predictions.extend(outputs.cpu().detach().numpy() )
            actual.extend(y.cpu().detach().numpy())
    return predictions, actual


def get_details(actual,predictions,train_dataset):
    actual_label = np.argmax(np.array(actual),axis=1)
    predicted_label = np.argmax(predictions, axis =1)
    accuracy = (actual_label == predicted_label).sum() / len(actual_label)
    number_of_classes = len(train_dataset.label_to_index)
    
    details = {
        'result_per_label': {i:0 for i in range(number_of_classes)},#how much i classiy for this class
        'total_per_label':  {i:0 for i in range(number_of_classes)}#how much the total that belong to this class
    }
    for i in range(len(actual_label)):
        result = 1 if actual_label[i] == predicted_label[i] else 0
        class_index = actual_label[i]
        details['result_per_label'][class_index]+= result
        details['total_per_label'][class_index]+= 1
    index_to_label = {val:key for key,val in train_dataset.label_to_index.items()}
    deta = {}
    for i in range(number_of_classes):
        if details['total_per_label'][i] ==0 or details['result_per_label'][i] ==0 :
            print('index: ',i,' Gloss: ',index_to_label[i],' accuracy: 0')
        else:
            print('index: ',i,' Gloss: ',index_to_label[i],' accuracy: ',(details['result_per_label'][i]/details['total_per_label'][i]))
            deta [i] = 'index: ' +str(i)+ ' Gloss: '+ str(index_to_label[i])+' accuracy: '+ str(details['result_per_label'][i]/details['total_per_label'][i])
    return accuracy, deta
    
def train(model,classification_loss_fn,epoch,train_loader,test_loader,scheduler,test_accuracy,CHECKPOINT_PATH,optimizer,device,train_dataset):
    max_seconds = 10.5*60*60  # 10 hours and 30 minutes
    start_time = time.time()
    if os.path.exists(os.path.join(CHECKPOINT_PATH,'checkpoint.pth')):
        checkpointsaver = torch.load(os.path.join(CHECKPOINT_PATH,'checkpoint.pth'), map_location='cpu')
        with open(os.path.join(CHECKPOINT_PATH,'data.json'), 'r') as file:
            metadatasaver = json.load(file)
    else:
        checkpointsaver = {}
        metadatasaver = {}

    total_loss = []
    glosses_details =[]
    current_best_accuracy = 0
    for ep in range(epoch,epoch+1000):
        model.train()
        total_loss_per_epoch = 0
        accuarcy = 0
        total_samples= 0
        for i, (x, croped_images, y) in enumerate(train_loader):
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                print(f"\n⏱️ Training loop exceeded 9 hours 30 minutes. Breaking at batch {i}.")
                return None

            optimizer.zero_grad()
        
            x = x.float().to(device)
            croped_images = croped_images.float().to(device)
        
            if y.ndim > 1:  # one-hot -> indices
                y = torch.argmax(y, dim=1)
        
            y = y.long().to(device)
        
            classification_output = model(x, croped_images)  # shape: [B, 2]
        
            # Compute loss
            classification_loss = classification_loss_fn(classification_output, y)
            loss = classification_loss
            total_loss_per_epoch += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
            # Compute accuracy
            predictions = torch.argmax(classification_output, dim=1)
            accuarcy += (predictions == y).sum().item()
            total_samples += y.size(0)
        
            print(f"\rEpoch {ep+1}, loss: {(total_loss_per_epoch/(i+1)):.8f}, [{i}/{len(train_loader)}]", end="")

        print(f"\rEpoch {ep+1}, loss: {(total_loss_per_epoch/len(train_loader)):.8f}")
        print()
        print("train accuracy is: ",(accuarcy/total_samples))
        avg_loss = total_loss_per_epoch/len(train_loader)
        total_loss.append(avg_loss)
        actual,predictions = predict(model,test_loader,device)
        acc, deta = get_details(actual,predictions,train_dataset)
        
        
        

        
        checkpointsaver['model_state_dict'] =  model.state_dict()
        checkpointsaver['optimizer_state_dict'] =  optimizer.state_dict()
        checkpointsaver['scheduler_state_dict'] =  scheduler.state_dict()
        checkpointsaver["epoch"] =  ep+1

        
         
        metadatasaver[ "epoch"]= ep
        metadatasaver[ "total train accuracy"]=(accuarcy/total_samples)
        metadatasaver[ "total test accuracy"]= acc
        metadatasaver["glosses test accuracy"]= deta
        

        
        if acc >= test_accuracy:
            test_accuracy =acc
            metadatasaver['best test accuracy'] = test_accuracy
            metadatasaver['best test accuracy details'] = deta
            checkpointsaver['best_model_state_dict']=model.state_dict()
            
        print('total accuracy: ',acc)
        with open("data.json", "w") as f:
            json.dump(metadatasaver, f,indent=4)
        print("best test accuracy: ",test_accuracy)


        torch.save(checkpointsaver, "checkpoint.pth")

        scheduler.step(avg_loss)
        print("\n///////////current learning rate: ",optimizer.param_groups[0]['lr']," /////////////\n")

    return total_loss

