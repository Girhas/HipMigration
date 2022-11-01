%%
% training datastores
load('labels/768x512_landmark_labels_1.mat');
%trainingData = objectDetectorTrainingData(gTruth,'SamplingFactor',1,'WriteLocation','TrainingData');

[imds,blds] = objectDetectorTrainingData(gTruth);
%[imds,blds] = objectDetectorTrainingData(foodImds);
cds = combine(imds,blds); % combine datastore
basenet = resnet50;
analyzeNetwork(basenet)
input_size = [224 224 3];
layerName = basenet.Layers(1).Name;
newinputLayer = imageInputLayer(input_size,'Name','newinput','Normalization','none');
lgraph = layerGraph(basenet);
lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');
lgraph = replaceLayer(lgraph,layerName,newinputLayer);
dlnet = dlnetwork(lgraph);
featureExtractionLayers = ["activation_22_relu","activation_25_relu","activation_28_relu","activation_31_relu","activation_34_relu","activation_37_relu","activation_40_relu"];
Anchors = {[80,60];...
            [100,40];...
            [80,60];...
            [80,60];... 
            [100,40];... 
            [80,60];...
            [40,30]};
anchorBoxes = {[122,177;223,84;80,94];...
               [111,38;33,47;37,18]};
classes = {'LeftForamen','LeftLowerIschium','LeftSciaticNotch',...
    'PubicSymph','RightForamen','RightLowerIschium','RightSciaticNotch'};
detector = yolov4ObjectDetector(dlnet,classes,Anchors);
disp(detector) 
analyzeNetwork(detector.Network)