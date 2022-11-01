% training datastores
load('labels/768x512_landmark_labels_1.mat');
%trainingData = objectDetectorTrainingData(gTruth,'SamplingFactor',1,'WriteLocation','TrainingData');

[imds,blds] = objectDetectorTrainingData(gTruth);
%[imds,blds] = objectDetectorTrainingData(foodImds);
cds = combine(imds,blds); % combine datastore
imds.Labels = blds.LabelData;