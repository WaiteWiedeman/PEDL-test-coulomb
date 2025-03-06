%% clear workspace
close all;
clear; 
clc;

%% parameters
sysParams = params_system();
ctrlParams = params_control();
trainParams = params_training();
trainParams.numSamples = 400;
trainParams.type = "pinn4"; % "dnn4","lstm4","pinn4","dnn6", "lstm6","pinn6"
trainParams.numLayers = 4;
trainParams.numNeurons = 256;
modelFile = "model\"+trainParams.type+"_"+num2str(trainParams.numLayers)+"_"+num2str(trainParams.numNeurons)+".mat";

%% Run simulation
f1 = 20; % initial force input
tSpan = [0,5]; % [0,5] 0:0.01:5
ctrlParams.fMax = [f1; 0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
plot_states(y(:,1),y(:,2:7));
plot_forces(y(:,1),y(:,8),y(:,10));

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

%% functions
function plot_forces(t,f1,fc)
    figure('Position',[500,100,800,800]);
    plot(t,f1,'k-',t,fc,'b-','LineWidth',2);
    legend("F1","Fc");
end

function plot_states(t,x)
    refClr = "blue";
    predClr = "red";
    labels= ["$q_1$","$q_2$","$\dot{q}_1$","$\dot{q}_2$","$\ddot{q}_1$","$\ddot{q}_2$"];
    figure('Position',[500,100,800,800]);
    tiledlayout("vertical","TileSpacing","tight")
    numState = size(x);
    numState = numState(2);
    for i = 1:numState
        nexttile
        plot(t,x(:,i),'Color',refClr,'LineWidth',2);
        hold on
        xline(1,'k--','LineWidth',2);
        ylabel(labels(i),"Interpreter","latex");
        if i == numState
            xlabel("Time (s)");
        end
        set(get(gca,'ylabel'),'rotation',0);
        set(gca, 'FontSize', 15);
        set(gca, 'FontName', 'Arial');
    end 
end
