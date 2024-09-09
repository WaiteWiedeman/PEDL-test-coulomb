%% clear workspace
close all; clear; clc;

%% test variables
file = "best_dnn4_models_2";
%file = "best_dnn_models_10";
%file = "best_pinn6_models";
% file = "best_pinn4_models";
%file = "best_lstm6_models";
% file = "best_lstm4_models";
net = load(file).dnn4_128_10_800;
%net = load(file).dnn_256_8_800;
%net = load(file).pinn6_256_8_800.trainedNet;
% net = load(file).pinn4_256_8_400.trainedNet;
%net = load(file).lstm6_256_8_400;
% net = load(file).lstm4_256_8_400;
sysParams = params_system();
ctrlParams = params_control();
trainParams = params_training();
trainParams.type = "dnn4"; % "dnn4","lstm4","pinn4","dnn6", "lstm6","pinn6"
numTime = 100;
f1Max = 15;
tSpan = 0:0.01:10; % [0,5] 0:0.01:5
predInterval = 10; 

%% simulation 
ctrlParams.fMax = [f1Max; 0];
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
