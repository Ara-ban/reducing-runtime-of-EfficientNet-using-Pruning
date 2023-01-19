# reducing-runtime-of-EfficientNet-using-Pruning

Multi-task EfficientNet-B2 realises the current state of art in facial expression and emotion recognition, the model excecution time however restricts interactive applications of such technology.
Working with the startup Emobot we were tasked to study reducing the runtime of the model and trying to keep the most accuracy possible
using weight pruning runtime was cut by a half and accuracy dropped from 62% to 58%, tested on a sample of 4000 images of the affectnet database

here I put the 5 files that allows testing the model before and after pruning

To test these files the folder Models should be downloaded from this repo https://github.com/HSE-asavchenko/face-emotion-recognition (shoutout to their work)

A sample of the affectnet database must also be downloaded and put on a folder named "DataAffectNet" or change the name of some paths in the files.
You can find one here https://www.kaggle.com/datasets/mouadriali/affectnetsample
