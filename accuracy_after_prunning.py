from matplotlib.image import imread
from facial_emotions import HSEmotionRecognizer
import pandas as pd

model_name = 'enet_b0_8_best_vgaf'
fer = HSEmotionRecognizer(model_name=model_name, device='cpu')

csv = pd.read_csv("face-emotion-recognition-main/DataAffectNet/valid-sample-affectnet.csv")
# print(csv.head())


emotions = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

accuracy = 0
nb_images = 4000

for j in range(nb_images):
    emotion = emotions[int(csv.loc[j, "emotion"])-1]
    loc_img = "face-emotion-recognition-main/DataAffectNet"+csv.loc[j, "image"][24:]

    face_img = imread(loc_img)
    predicted_emotion, scores = fer.predict_emotions(face_img, logits=True)

    if predicted_emotion == emotion:
        accuracy += 1

print("Accuracy : ", 100*accuracy/nb_images, "%")
