%%
close all;
clear; 
clc;

%% set task type
params = parameters();
tSpan = 0:0.01:10;
tRMSE = 500; % time steps not in rmse calculation
tForceStop = 1;
ctrlOptions = control_options();

ds = load('trainingData.mat');
numSamples = size(ds.samples,1);
modelFile = "best_lstm_models.mat";
maxEpochs = 50;
F1Min = max(20,params(10));
Fmax = 15;

%% Test 1
net = load(modelFile).best_train_RMSE;
seqSteps = 20;
ctrlOptions.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
startIdx = initIdx-seqSteps+1;
state = {[t(startIdx:initIdx),x(startIdx:initIdx,:)]'};
x0 = arrayDatastore(state,'OutputType',"same",'ReadSize',200);
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp) 
    xp(i,:) = predict(net,combine(x0, arrayDatastore(tp(i)-t0,'ReadSize',200)));
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best training RMSE", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlOptions.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% Test 2
net = load(modelFile).best_train_loss;
seqSteps = 20;
ctrlOptions.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
startIdx = initIdx-seqSteps+1;
state = {[t(startIdx:initIdx),x(startIdx:initIdx,:)]'};
x0 = arrayDatastore(state,'OutputType',"same",'ReadSize',200);
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp) 
    xp(i,:) = predict(net,combine(x0, arrayDatastore(tp(i)-t0,'ReadSize',200)));
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best training loss", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlOptions.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% Test 3
net = load(modelFile).best_val_RMSE;
seqSteps = 20;
ctrlOptions.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
startIdx = initIdx-seqSteps+1;
state = {[t(startIdx:initIdx),x(startIdx:initIdx,:)]'};
x0 = arrayDatastore(state,'OutputType',"same",'ReadSize',200);
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp) 
    xp(i,:) = predict(net,combine(x0, arrayDatastore(tp(i)-t0,'ReadSize',200)));
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best validation RMSE", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlOptions.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% Test 4
net = load(modelFile).best_val_loss;
seqSteps = 10;
ctrlOptions.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
startIdx = initIdx-seqSteps+1;
state = {[t(startIdx:initIdx),x(startIdx:initIdx,:)]'};
x0 = arrayDatastore(state,'OutputType',"same",'ReadSize',200);
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp) 
    xp(i,:) = predict(net,combine(x0, arrayDatastore(tp(i)-t0,'ReadSize',200)));
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best validation loss", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlOptions.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)

%% Test 5
net = load(modelFile).best_model_eval;
seqSteps = 20;
ctrlOptions.fMax = [F1Min+Fmax;0];
y = sdpm_simulation(tSpan,ctrlOptions);
t = y(:,1);
x = y(:,4:9);
initIdx = find(t >= tForceStop,1,'first');
t0 = t(initIdx);
startIdx = initIdx-seqSteps+1;
state = {[t(startIdx:initIdx),x(startIdx:initIdx,:)]'};
x0 = arrayDatastore(state,'OutputType',"same",'ReadSize',200);
tp = t(initIdx+1:end);
xp = zeros(length(tp),6);
for i = 1:length(tp) 
    xp(i,:) = predict(net,combine(x0, arrayDatastore(tp(i)-t0,'ReadSize',200)));
end
rmse = root_square_err(1:length(xp)-tRMSE,x(initIdx+1:end,:),xp);
titletext = {"best test accuracy", "Test RMSE through 5s: " + num2str(mean(rmse,"all")), "Force Input: " + num2str(ctrlOptions.fMax(1)) + " N"};
plot_compared_states(t,x,tp,xp,titletext)
