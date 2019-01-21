clf,clc,close all
% The code plots the results when changing the minimum number of slices
% taken from the test meters. Effectively changing the test set to only
% include "good" meters.

%EXCLUDING HOT WATER
x = [1,6,11,16,21,26,31,36,41,46];
aDNN = [71, 74.6,75.9,77.3, 78.2,79.0,79.8,80.1,80.6,81.2];
num_meters = [224,201,191,185,179,176,173,171,170,165];
aBNN = [69.6,74.1,76.4,77.8,78.8,79.5,80.3,80.7,80.6,81.2];
% ALL CLASSSES
yyaxis left
plot(x,aDNN)
hold on
plot(x,aBNN,'r')
ylabel(" Final Test Accuracy [%] ")


yyaxis right
plot(x,num_meters)
ylim([160 250])
ylabel(" Number of Meters in Test Set ")
xlabel(" Min number of Slices/Meter ")
lgd  = legend("DNN","BNN");
lgd.Location = 'northwest';

figure
a2DNN = [62.5,65.2,65.6,66.2,67.5,67.5,68.9,69.1,69.1,69.9];
%x = [1,6,11,16,21,26,31,36,41,46];
num_meters2 = [240,221,212,207,203,200,196,194,191,186];
a2BNN = [62.1,64.7,65.6,66.7,67.5,67.5,68.4,68.0,68.1,69.4];
yyaxis left
plot(x,a2DNN)
hold on
plot(x,a2BNN,'r')
ylabel(" Final Test Accuracy [%] ");


yyaxis right
plot(x,num_meters2)
ylim([180 250]);
ylabel(" Number of Meters in Test Set ");
xlabel(" Min number of Slices/Meter ");
lgd  = legend("DNN","BNN");
lgd.Location = 'northwest';








%%% Plotting chaNGE OF MAX
x3 = 1:3:58;
p3DNN = [62.7,66.1,65.7,64.8,65.3,64.4,64.4,66.5,66.5,66.5,68.6,68.6,68.2,69.5,69.9,69.5,69.5,69.5,69.5,69.9];
p3BNN = [59.7,66.5,69.1,68.2,68.2,69.9,69.1,69.1,68.6,69.5,69.5,68.2,69.5,69.5,68.6,69.1,67.8,68.6,69.1,69.1];
figure
plot(x3,p3DNN)
hold on
plot(x3,p3BNN,'r')
xlabel("Max number of Slices/Meter")
ylabel("Final Test Accuracy [%]")
lgd = legend("DNN","BNN");
lgd.Location='east';
