function sdpm_snapshot(sysParams, t, x, th, xp, thp, t_snap)
indices = find(t <= t_snap, 1, 'last');
Xcg = x(indices);
theta = th(indices);
Xcg_pred = xp(indices);
theta_pred = thp(indices);

% Animation
Ycg = 0.5;
% plot limits
Xmin = -10;
Xmax = 10;
Ymin = -0.5;
Ymax = 2;
l = sysParams.L; % pendulum rod length

f = figure;
f.Position = [400 400 1000 600];

subplot(3,1,1)
plot(t,x,'g',t,xp,'r--','LineWidth',2)
ylabel("$q_1$","Interpreter","latex");
xline(t_snap,'k--','Time in image', 'LineWidth',1);
grid on
axis([0 max(t) min(xp)-1 max(xp)+1])

subplot(3,1,2)
plot(t,th,'b',t,thp,'r--','LineWidth',2)
ylabel("$q_2$","Interpreter","latex");
xline(t_snap,'k--','Time in image', 'LineWidth',1);
grid on
axis([0 max(t) min(thp)-1 max(thp)+1])

subplot(3,1,3)
hold on
% Plot one frame...
% True system
patch(Xcg+[-1 1 1 -1] ,Ycg+[0.5 0.5 -0.5 -0.5],'g') % cart
plot([-10 -10 20],[5 0 0],'k','LineWidth',4) % ground and wall.
plot([-10, -8, -8:((9 +Xcg-4)/9):Xcg-3, Xcg-3, Xcg-1],...
    Ycg+0.25+[0 0 0 .125 -.125 .125 -.125 .125 -.125 .125 -.125 0 0 0],'r','LineWidth',2) % spring
plot([-10 -8],Ycg-.25+[0 0],'b',[-8 -8],Ycg-.25+[.125 -.125],'b',...
    [Xcg-15, Xcg-3,Xcg-3,Xcg-15,],Ycg-.25+[.15 .15 -.15 -.15],'b',...
    [ Xcg-3,Xcg-1],Ycg-.25+[0 0],'b','LineWidth',2)    %  damper
plot([Xcg Xcg+l*sin(theta)],[Ycg Ycg-l*cos(theta)],'k','LineWidth',3); % plots rod
plot(Xcg+l*sin(theta),Ycg-l*cos(theta),'Marker','o','MarkerSize',4,'MarkerFaceColor','m','MarkerEdgeColor','m'); % plots bob

% System predicted by model
patch(Xcg_pred+[-1 1 1 -1] ,Ycg+[0.5 0.5 -0.5 -0.5],'g','FaceAlpha',0,'LineStyle',"--"); % cart
plot([Xcg_pred Xcg_pred+l*sin(theta_pred)],[Ycg Ycg-l*cos(theta_pred)],'k','LineWidth',3,'LineStyle','--'); % plots rod
plot(Xcg_pred+l*sin(theta_pred),Ycg-l*cos(theta_pred),'Marker','o','MarkerSize',4,'MarkerEdgeColor','k'); % plots bob 'MarkerFaceColor','k',

axis([Xmin Xmax Ymin Ymax])
daspect([1 1 1])
titletext = {"Spring Damper Pendulum Mass system at "+num2str(t_snap)+" s"};
title(titletext)

saveas(f,'sdpm_snapshot.jpg')
