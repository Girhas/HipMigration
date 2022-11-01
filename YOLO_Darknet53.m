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

lgraph = replaceLayer(lgraph,'input',inputLayer);
%lgraph = replaceLayer(lgraph,learnableLayer,newLearnableLayer);
%newClassLayer = classificationLayer('Name','new_classoutput');
%lgraph = replaceLayer(lgraph,classLayer,newClassLayer);
lgraph = removeLayers(lgraph,["output"]);
analyzeNetwork(lgraph);
dlnet = dlnetwork(lgraph);
Anchors = {[80,60];...
            [100,40];...
            [80,60];...
            [80,60];... 
            [100,40];... 
            [80,60];...
            [40,30]};

classes = {'LeftForamen','LeftLowerIschium','LeftSciaticNotch',...
    'PubicSymph','RightForamen','RightLowerIschium','RightSciaticNotch'};
lgraph = yolov4ObjectDetector(dlnet,classes,Anchors);
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



%% 
a = dir(['./images/768x512/testing/' '*.tiff']);
n = numel(a);

for i = 1:n
    img = imread(['./images/768x512/testing/pelvis_00', num2str(i), '.tiff']);    
    [bboxes,scores,labels] = detect(THAnet_768x512,img  );
    if(~isempty(bboxes))
                
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
end