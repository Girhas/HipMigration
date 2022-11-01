load('labels/768x512_landmark_labels_1.mat');
%trainingData = objectDetectorTrainingData(gTruth,'SamplingFactor',1,'WriteLocation','TrainingData');

[imds,blds] = objectDetectorTrainingData(gTruth);
%[imds,blds] = objectDetectorTrainingData(foodImds);
cds = combine(imds,blds); % combine datastore

input_size = [768 768 1];
inputLayer = imageInputLayer(input_size,'Name','input','Normalization','none');
filterSize = [3 3];
net = resnet101;
analyzeNetwork(net)
net.Layers(1)
imds.Labels = cds.UnderlyingDatastores{1,2}.LabelData;

lgraph = layerGraph(net);
learnableLayer = lgraph.Layers(345);
classLayer = lgraph.Layers(347);
numClasses = 7;

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','newclassoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore([224 224],imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore([224 224],imdsValidation);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.002, ...
        'LearnRateSchedule','none', ...
        'LearnRateDropPeriod',20, ... 
        'LearnRateDropFactor',0.5, ...
        'Verbose',true, ...
        'VerboseFrequency', 70, ...
        'MiniBatchSize',2, ...
        'MaxEpochs',60,...
        'Shuffle','every-epoch', ...
        'DispatchInBackground',false,...
        'ExecutionEnvironment','gpu', ...        
        'Plots','training-progress');
net = trainNetwork(cds,lgraph,options);
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)


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
