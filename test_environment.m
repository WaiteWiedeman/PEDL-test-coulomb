%% clear workspace
close all; clear; clc;

%% test variables
file = "best_dnn4_models";
%file = "best_pinn_models_3";
net = load(file).dnn4_128_6_400;
%net = load(file).pinn_128_10_800.trainedNet;
sysParams = params_system();
ctrlParams = params_control();
trainParams = params_training();
trainParams.type = "dnn4"; % "dnn4","lstm4","pinn4","dnn6", "lstm6","pinn6"
numTime = 100;
f1Min = 5;
f1Range = 10;
tSpan = [0,10];
predInterval = 10;

%% simulation
f1Max = f1Min+f1Range; 
ctrlParams.fMax = [f1Max; 0];
y = sdpm_simulation(tSpan, sysParams, ctrlParams);
t = y(:,1);
x = y(:,2:7);
[xp, rmseErr, refTime] = evaluate_single(net, t, x, ctrlParams, trainParams, tSpan, predInterval, numTime, trainParams.type);
plot_compared_states(t,x,t,xp)
disp(mean(rmseErr,'all'))
