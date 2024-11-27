%% 
close all;
clear; 
clc;
sysParams = params_system();
ctrlParams = params_control();
trainParams = params_training();
trainParams.numSamples = 400;
trainParams.type = "pinn4"; % "dnn4","lstm4","pinn4","dnn6", "lstm6","pinn6"
trainParams.numLayers = 4;
trainParams.numNeurons = 256;
modelFile = "model\"+trainParams.type+"_"+num2str(trainParams.numLayers)+"_"+num2str(trainParams.numNeurons)+".mat";

%% generate samples
if ~exist("\data\", 'dir')
   mkdir("data");
end
f1Max = [15,35];
tSpan = [0,5];
[dataFile, fMaxRange] = generate_samples(sysParams, ctrlParams, trainParams, f1Max, tSpan);
% plot(sort(fMaxRange));
% histogram(sort(fMaxRange),trainParams.numSamples)

%% train model
if ~exist("\model\", 'dir')
   mkdir("model");
end
switch trainParams.type
    case "dnn4"
        [xTrain,yTrain,layers,options] = train_dnn_model_4(dataFile, trainParams);
        [net,info] = trainNetwork(xTrain,yTrain,layers,options);
        plot(layers)
    case "lstm4"
        [xTrain,yTrain,layers,options] = train_lstm_model_4(dataFile, trainParams);
        [net,info] = trainNetwork(xTrain,yTrain,layers,options);
        plot(layers)
    case "pinn4"
        monitor = trainingProgressMonitor;
        output = train_pinn_model_4(dataFile, trainParams,sysParams,ctrlParams,monitor);
        net = output.trainedNet;
    case "dnn6"
        [net,info] = trainNetwork(xTrain,yTrain,layers,options);
        [xTrain,yTrain,layers,options] = train_dnn_model_6(dataFile, trainParams);
        plot(layers)
    case "lstm6"
        [net,info] = trainNetwork(xTrain,yTrain,layers,options);
        [xTrain,yTrain,layers,options] = train_lstm_model_6(dataFile, trainParams);
        plot(layers)
    case "pinn6"
        monitor = trainingProgressMonitor;
        output = train_pinn_model_4(dataFile, trainParams,sysParams,ctrlParams,monitor);
        net = output.trainedNet;
    otherwise
        disp("unspecified type of model.")
end

% training with numeric array data
trainLoss = info.TrainingLoss;
save(modelFile, 'net');
% disp(info)

%% simulation 
f1 = 20; % initial force input
numTime = 100;
tSpan = [0,5]; % [0,5] 0:0.01:5
predInterval = tSpan(2); 
ctrlParams.fMax = [f1; 0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
t = y(:,1);
x = y(:,2:7);
[xp, rmseErr, refTime] = evaluate_single(net, t, x, ctrlParams, trainParams, tSpan, predInterval, numTime, trainParams.type);
plot_compared_states(t,x,t,xp)
sdpm_snapshot(sysParams, t, x(:,1), x(:,2), xp(:,1), xp(:,2), 3)
sdpm_animation(sysParams, t, x(:,1), x(:,2), xp(:,1), xp(:,2))
disp(mean(rmseErr,'all'))

%% evaluate for four states
f1Max = [5,30];
tSpan = [0,5];
predIntervel = 5;
numCase = 50;
numTime = 100;
%avgErr = evaluate_model(net, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, numCase, numTime, trainParams.type);
avgErr = evaluate_model_with_4_states(net, sysParams, ctrlParams, trainParams, f1Max, tSpan, predInterval, numCase, numTime, trainParams.type);
disp(avgErr)
