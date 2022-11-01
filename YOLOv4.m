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

input_size = [768 768 1];


%%  
% layer creation
inputLayer = imageInputLayer(input_size,'Name','input','Normalization','none');
filterSize = [3 3];

middleLayers = [
    convolution2dLayer(filterSize, 128, 'Padding', 2,'Name','conv_1',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(8, 'Stride',8,'Name','maxpool1')
    convolution2dLayer(filterSize, 128, 'Padding', 2,'Name', 'conv_2',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
   
    maxPooling2dLayer(4, 'Stride',4,'Name','maxpool2')
    convolution2dLayer(filterSize, 128, 'Padding', 2,'Name','conv_3',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
     
    maxPooling2dLayer(4, 'Stride',4,'Name','maxpool3')
    convolution2dLayer(filterSize, 12, 'Padding', 2,'Name','conv_4',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu_4')

    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool4')
    convolution2dLayer(filterSize, 128, 'Padding', 1,'Name','conv_5',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN5')
    reluLayer('Name','relu_5')

    maxPooling2dLayer(1, 'Stride',1,'Name','maxpool5')
    convolution2dLayer(filterSize, 64, 'Padding', 1,'Name','conv_6',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN6')
    reluLayer('Name','relu_6')

    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool6')
    convolution2dLayer(filterSize, 64, 'Padding', 1,'Name','conv_7',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN7')
    reluLayer('Name','relu_7')

    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool7')
    convolution2dLayer(filterSize, 32, 'Padding', 1,'Name','conv_8',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN8')
    reluLayer('Name','relu_8')

    maxPooling2dLayer(1, 'Stride',1,'Name','maxpool8')
    convolution2dLayer(filterSize, 128, 'Padding', 1,'Name','conv_9',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN9')
    reluLayer('Name','relu_9')

    maxPooling2dLayer(1, 'Stride',1,'Name','maxpool9')
    convolution2dLayer(filterSize, 128, 'Padding', 1,'Name','conv_10',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN10')
    reluLayer('Name','relu_10')

    maxPooling2dLayer(1, 'Stride',1,'Name','maxpool10')
    convolution2dLayer(filterSize, 128, 'Padding', 1,'Name','conv_11',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN11')
    reluLayer('Name','relu_11')

    maxPooling2dLayer(1, 'Stride',1,'Name','maxpool11')
    convolution2dLayer(filterSize, 12, 'Padding', 1,'Name','conv_12',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN12')
    reluLayer('Name','relu_12')

    ];

%%
lgraph = layerGraph([inputLayer; middleLayers]);
% analyzeNetwork(lgraph);
numClasses = 7;

Anchors = {[80,60];...
            [100,40];...
            [80,60];...
            [80,60];... 
            [100,40];... 
            [80,60];...
            [40,30]};
anchorBoxes = {[122,177;223,84;80,94];...
               [111,38;33,47;37,18]};
AnchorBoxes = [{80, 60};{100, 40}; {80, 60}; {80, 60}; {100, 40}; {80, 60}; {40, 30}];
dlnet = dlnetwork(lgraph);
featureExtractionLayers = {'relu_5','relu_6','relu_7','relu_8','relu_9','relu_10','relu_11'};
classes = {'LeftForamen','LeftLowerIschium','LeftSciaticNotch',...
    'PubicSymph','RightForamen','RightLowerIschium','RightSciaticNotch'};
lgraph = yolov4ObjectDetector(dlnet,classes,Anchors,DetectionNetworkSource=featureExtractionLayers);
analyzeNetwork(lgraph.Network);
%lgraph = yolov4Layers(input_size,numClasses,Anchors,dlnet,'relu_4');



%%
% options = trainingOptions('sgdm', ...
%         InitialLearnRate=0.002, ...
%         LearnRateSchedule='none', ...
%         LearnRateDropPeriod=,20, ... 
%         LearnRateDropFactor=0.5, ...
%         Verbose=true, ...
%         VerboseFrequency= 70, ...
%         MiniBatchSize=2, ...
%         MaxEpochs=60,...
%         Shuffle='every-epoch', ...
%         DispatchInBackground=false,...
%         ExecutionEnvironment='gpu');
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.002, ...
    MiniBatchSize=2,...
    MaxEpochs=1, ...
    Verbose = true,...
    DispatchInBackground=false,...
    LearnRateSchedule="none",...
    LearnRateDropPeriod=20,...
    LearnRateDropFactor=0.5,...
    BatchNormalizationStatistics="moving",...
    CheckpointPath='nets/yolov4Checkpoint',...
    ResetInputNormalization=false,...
    ExecutionEnvironment='gpu',...
    VerboseFrequency=70);

% options = trainingOptions('sgdm', ...
%         'InitialLearnRate',0.002, ...
%         'LearnRateSchedule','none', ...
%         'LearnRateDropPeriod',20, ... 
%         'LearnRateDropFactor',0.5, ...
%         'Verbose',true, ...
%         'VerboseFrequency', 70, ...
%         'MiniBatchSize',2, ...
%         'MaxEpochs',60,...
%         'Shuffle','every-epoch', ...
%         'DispatchInBackground',false,...
%         'ExecutionEnvironment','gpu', ...        
%         'Plots','training-progress');
%%
% train the detector
[THAnet_768x512v4, info]= trainYOLOv4ObjectDetector(cds,lgraph,options);



%% 
a = dir(['./images/768x512/testing/' '*.tiff']);
n = numel(a)

for i = 1:n
    img = imread(['./images/768x512/testing/pelvis_00', num2str(i), '.tiff']);
    imshow(img)
    [bboxes,scores,labels] = detect(THAnet_768x512v4,img);                
        [c,r] = size(scores);
        label_str = cell(c,1);
        my_label = cell(c,1);
        for ii=1:c
            label_str{ii} = [];
            my_label{ii} = [];
            label_str{ii} = [cellstr(labels(ii)), ': ', num2str(scores(ii)*100,'%0.2f'), '%'];            
            [cc,rr] = size(label_str{ii});
            for jj=1:rr
                my_label{ii} = strcat(my_label{ii},label_str{ii}(1,jj));
            end
            label_str{ii} = char( my_label{ii});
        end                
        
        img_a = insertObjectAnnotation(img,'rectangle',bboxes, label_str,'TextBoxOpacity',0.25,'FontSize',10);
        % img_a = insertObjectAnnotation(img,'rectangle',bboxes, labels,'TextBoxOpacity',0.25,'FontSize',10);
        
        figure
        imshow(img_a)
end