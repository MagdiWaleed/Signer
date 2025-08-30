import torch 
import torch.nn as nn
from torchvision import models

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.pos_embed = nn.Embedding(32, 2048)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        pos_embed = self.pos_embed(pos)
        return x + pos_embed



class CrossAttention(nn.Module):
    def __init__(self, image_dim, pose_dim, project_dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.image_pre_norm = nn.LayerNorm(normalized_shape=image_dim)
        self.pose_pre_norm = nn.LayerNorm(normalized_shape=pose_dim)

        self.pose_project = nn.Linear(pose_dim, project_dim)
        self.image_project = nn.Linear(image_dim, project_dim)

        self.image_self_attention= nn.MultiheadAttention(embed_dim=project_dim, num_heads=num_heads, batch_first=True)
        self.pose_self_Attenstion = nn.MultiheadAttention(embed_dim=project_dim, num_heads=num_heads, batch_first=True)

        self.image_layer_norm = nn.LayerNorm(normalized_shape=project_dim)
        self.pose_layer_norm = nn.LayerNorm(normalized_shape=project_dim)
        # self.output_layer_norm = nn.LayerNorm(normalized_shape=int(project_dim*2))


        self.cross_attention1 = nn.MultiheadAttention(embed_dim=project_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention2 = nn.MultiheadAttention(embed_dim=project_dim, num_heads=num_heads, batch_first=True)


    def forward(self, image_input, pose_input):
        image_features = self.image_pre_norm(image_input)
        pose_features = self.pose_pre_norm(pose_input)

        pose_projection = self.pose_project(pose_features)
        image_projection = self.image_project(image_features)

        pose_projection_attention, _ = self.pose_self_Attenstion(query=pose_projection, key=pose_projection, value=pose_projection)
        image_projection_attention, _ = self.image_self_attention(query=image_projection, key=image_projection, value=image_projection)

        residual_image_connection = image_projection+image_projection_attention
        residual_pose_connection = pose_projection+pose_projection_attention

        image = self.image_layer_norm(residual_image_connection)
        pose = self.pose_layer_norm(residual_pose_connection)

        attn_output1, _ = self.cross_attention1(query=image, key=pose, value=pose)
        attn_output2, _ = self.cross_attention2(query=pose, key=image, value=image)

        # output = self.final_feedforward_layer(torch.cat([ attn_output1, attn_output2 ],dim=2))

        # output = self.output_layer_norm(output)

        return torch.cat([ attn_output1, attn_output2 ],dim=2)

class Model(nn.Module):
    def __init__(self,embedding = 5):
        super().__init__()

        self.crossAttentionLayer = CrossAttention(image_dim=6320, pose_dim=3102,project_dim=1024, num_heads=4)


        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        mobilenet_features = mobilenet.features

        t=0
        for layer in mobilenet.parameters():
            t+=1

        for i,layer in enumerate(mobilenet.parameters()):
            layer.requires_grad = False
            if i== (t-4) or i== (t-5):
              layer.requires_grad = True

        self.cropedImageCNN = nn.Sequential(
                mobilenet_features,

                nn.Conv2d(1280, 1580, kernel_size=3),
                nn.BatchNorm2d(1580),
                nn.SiLU(),
                nn.AvgPool2d(kernel_size=2),

                nn.Conv2d(1580, 1580, kernel_size=3),
                nn.BatchNorm2d(1580),
                nn.SiLU(),
                nn.AvgPool2d(kernel_size=2),

                nn.Flatten()
            )


        self.leftHandCNN = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.AvgPool2d(2),       # -> 40x40

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AvgPool2d(2),       # -> 20x20

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AvgPool2d(2),       # -> 10x10

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AvgPool2d(2),       # -> 5x5
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1280),

        )

        self.rightHandCNN = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.AvgPool2d(2),       # -> 40x40

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AvgPool2d(2),       # -> 20x20

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AvgPool2d(2),       # -> 10x10

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AvgPool2d(2),       # -> 5x5
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1280),

        )

        self.leftHandVRegionEmbedding = nn.Sequential(
            nn.Linear(1,embedding),
        )
        self.leftHandHRegionEmbedding = nn.Sequential(
            nn.Linear(1,embedding),
        )
        self.rightHandVRegionEmbedding = nn.Sequential(
            nn.Linear(1,embedding),
        )
        self.rightHandHRegionEmbedding = nn.Sequential(
            nn.Linear(1,embedding),
        )
        self.leftHandDistance = nn.Sequential(
            nn.Linear(2,embedding),
        )
        self.rightHandDistance = nn.Sequential(
            nn.Linear(2,embedding),
        )
        self.landmarksEmbedding = nn.Sequential(
            nn.Linear(1106 ,512),
        )

        encoder_layer = nn.TransformerEncoderLayer(
                d_model=2048,
                nhead=8,
                dim_feedforward=3072,
                dropout=0.0,
                activation='gelu',
                batch_first=True
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)


        self.positionalEmbedding = PositionalEmbedding()

        self.classifier = nn.Sequential(
            nn.Linear(2048, 26),
        )
        self.landmarksLayerNorm = nn.LayerNorm(normalized_shape=1106)
        # self.dropout1 = nn.Dropout(0.05)
        # self.dropout2 = nn.Dropout(0.05)

    def forward(self,x,croped_image, train= False):
        batch_size,frames, features = x.size()

        # print(x.size())

        # print(x.size())

        LHI,RHI,LHVR,RHVR,LHHR,RHHR,LHD,RHD,ALL_LANDMARKS =x[:,:,:19200]/255,x[:,:,19200:38400]/255,x[:,:,38400],x[:,:,38401],x[:,:,38402],x[:,:,38403],x[:,:,38404:38406],x[:,:,38406:38408],x[:,:,38408:]
        LHI = LHI.reshape(batch_size,frames,80,80,3).float()
        LHI = LHI.reshape(batch_size*frames, 80,80,3)
        LHI = LHI.permute(0,3,1,2)
        LHI_output = self.leftHandCNN(LHI)
        LHI_output = LHI_output.reshape(batch_size,frames,-1) #1280

        RHI = RHI.reshape(batch_size,frames,80,80,3).float()
        RHI = RHI.reshape(batch_size*frames, 80,80,3)
        RHI = RHI.permute(0,3,1,2)
        # print(RHI.shape)
        RHI_output = self.rightHandCNN(RHI)
        RHI_output = RHI_output.reshape(batch_size,frames,-1) #1280

        # print('left hand regoin',LHR.shape)
        LHVR_output = self.leftHandVRegionEmbedding(LHVR.unsqueeze(-1).float())
        LHHR_output = self.leftHandHRegionEmbedding(LHHR.unsqueeze(-1).float())
        RHVR_output = self.rightHandVRegionEmbedding(RHVR.unsqueeze(-1).float())
        RHHR_output = self.rightHandHRegionEmbedding(RHHR.unsqueeze(-1).float())

        LHD_output = self.leftHandDistance(LHD.float())
        RHD_output = self.rightHandDistance(RHD.float())

        all_landmarks_output = self.landmarksEmbedding(self.landmarksLayerNorm(ALL_LANDMARKS.float()))

        batch_size,frames,c,w,h = croped_image.shape
        croped_images_input = croped_image.reshape(batch_size*frames,c,w,h)

        croped_output = self.cropedImageCNN(croped_images_input/255)#[128, 512, 3, 3]

        croped_output = croped_output.reshape(batch_size,frames,-1)
        # croped_output = self.dropout1(croped_output)
        # print(croped_output.shape)
        flattend_output = torch.cat([LHI_output, RHI_output,LHVR_output,LHHR_output, RHVR_output,RHHR_output,LHD_output, RHD_output, all_landmarks_output],dim=2)
        # flattend_output = self.dropout2(flattend_output)
        # flattend_output = self.dropout(flattend_output)
        combined_output = self.crossAttentionLayer(croped_output,flattend_output)
        combined_output = self.positionalEmbedding(combined_output)

        transformer_output = self.transformer(combined_output)

        global_average_pooling = transformer_output.mean(dim=1)


        output = self.classifier(global_average_pooling)


        return output
