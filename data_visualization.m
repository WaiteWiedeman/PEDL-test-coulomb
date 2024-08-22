%% clear workspace
close all;
clear;
clc;

%% plot data
ds = load('trainingData.mat');
ind = randi(length(ds.samples));
data = load(ds.samples{ind,1}).state;
y = data';
plot_states(y(:,1),y(:,4:9));
plot_forces(y(:,1),y(:,2),y(:,10));

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
