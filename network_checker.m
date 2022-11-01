%% create_THAnet
% this script generates a Deep Learning CNN network for Hip Bone Landmark
% region detection.

% specifically the network input layer is 768x768 
% the reader should explore the other network layers
% and the training options


%%
% training datastores
load('labels/768x512_landmark_labels_1.mat');
%trainingData = objectDetectorTrainingData(gTruth,'SamplingFactor',1,'WriteLocation','TrainingData');

[imds,blds] = objectDetectorTrainingData(gTruth);
%[imds,blds] = objectDetectorTrainingData(foodImds);
cds = combine(imds,blds); % combine datastore

input_size = [256 256 3];


%%  
% layer creation
inputLayer = imageInputLayer(input_size,'Name','newinput','Normalization','none');

%%
numClasses = 7;

%lgraph = darknet53("Weights","none");

%analyzeNetwork(lgraph)
net = darknet53;
lgraph = layerGraph(net);
% 
% learnableLayer = 'conv53';
%classLayer = "output";
% newLearnableLayer = convolution2dLayer(1,numClasses, ...
% 'Name','new_conv', ...
% 'WeightLearnRateFactor',10, ...
% 'BiasLearnRateFactor',10);

pool = maxPooling2dLayer(4, 'Stride',4,'Name','maxpool2');
conv =  convolution2dLayer([1 1], 1000, 'Padding', 'same','Name','conv_55',...
    'Stride',[1 1],'DilationFactor',[1 1]);
lgraph.Layers(182)

lgraph = replaceLayer(lgraph,'input',inputLayer);
lgraph = replaceLayer(lgraph,'avg1',pool);
lgraph = replaceLayer(lgraph,'conv53',conv);
%lgraph = replaceLayer(lgraph,learnableLayer,newLearnableLayer);
%newClassLayer = classificationLayer('Name','new_classoutput');
%lgraph = replaceLayer(lgraph,classLayer,newClassLayer);
lgraph = removeLayers(lgraph,["output","softmax"]);
analyzeNetwork(lgraph);
dlnet = dlnetwork(lgraph);
Anchors = {[80,60];...
            [100,40];...
            [80,60];...
            [80,60];... 
            [100,40];... 
            [80,60];...
            [40,30]};
featureExtractionLayers = {'leakyrelu39','leakyrelu41','leakyrelu43','leakyrelu46','leakyrelu48','leakyrelu50','leakyrelu52'};
classes = {'LeftForamen','LeftLowerIschium','LeftSciaticNotch',...
    'PubicSymph','RightForamen','RightLowerIschium','RightSciaticNotch'};
lgraph = yolov4ObjectDetector(dlnet,classes,Anchors,DetectionNetworkSource=featureExtractionLayers);
%lgraph = yolov4Layers(input_size,numClasses,Anchors,dlnet,'relu_4');



%%
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.002, ...
    MiniBatchSize=2,...
    MaxEpochs=60, ...
    BatchNormalizationStatistics="moving",...
    ResetInputNormalization=false,...
    ExecutionEnvironment='gpu',...
    VerboseFrequency=70);

%%
% train the detector
[THAnet_768x512, info] = trainYOLOv4ObjectDetector(cds,lgraph,options);