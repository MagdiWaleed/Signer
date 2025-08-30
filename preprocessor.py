from PreprocessingPipeline.Preprocess import creat_gloss_folder
from PreprocessingPipeline.Compressor import compressFolder

if __name__ == "__main__":
    creat_gloss_folder( dataframe_path ='/content/drive/MyDrive/Graduation Project (1)/Datasets/Australian Dataset/Data/train_labels.csv',
                data_path=  '/content/drive/MyDrive/Graduation Project (1)/Datasets/Australian Dataset/Data/train',
                output_folder ='/content/drive/MyDrive/Graduation Project (1)/Datasets/Australian Dataset/Data/CONTINUE/ORIGINAL_DATA',
                chooser=    [
                    "angel",
                    "you"
                ],
                RGB_Only=True)
    compressFolder(
    "/content/drive/MyDrive/Graduation Project (1)/Datasets/Australian Dataset/Data/CONTINUE",
            [

                    "angel",
                    "you"
           ]
    )   