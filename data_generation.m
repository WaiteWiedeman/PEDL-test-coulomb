%%
close all;
clear; 
clc;

%% Generate Data for Training 
% Mass-Spring-Damper-Pendulum Dynamics System Parameters
sysParams = params_system();
ctrlParams = params_control();
trainParams = params_training();
trainParams.numSamples = 1000;
if ~exist("\data\", 'dir')
    mkdir("data");
end
dataFile = "trainingSamples.mat";
tSpan = [0,5];
F1Range = 15;
% simulate and save data
samples = cell(trainParams.numSamples,1);
f1Min = 5;
for i = 1:trainParams.numSamples
    disp("generate data for " + num2str(i) + "th sample.");
    % random max force F1 for each sample in a varying range of 10N
    ctrlParams.fMax = [f1Min; 0]+rand(2,1).*[F1Range; 0];
    y = sdpm_simulation(tSpan, sysParams, ctrlParams);
    state = y';
    fname=['data\input',num2str(i),'.mat'];
    save(fname, 'state');
    samples{i} = fname;
end
% samples = reshape(samples, [], 1); % make it row-based
save(dataFile, 'samples');
plot_states(y(:,1),y(:,2:7));
plot_forces(y(:,1),y(:,8),y(:,10));

%% 
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