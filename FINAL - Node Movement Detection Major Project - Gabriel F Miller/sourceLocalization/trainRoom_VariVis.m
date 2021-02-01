%{
This program visulizes the mean and standard deviation of error in
localization of the SSGP process. To simulate said process, please see
trainRoom_VariSim.m
%}
 
% % load desired scenario (scroll past commented section) or uncomment and simulate new environment RTFs 
clear
addpath ./RIR-Generator-master
addpath ./functions
addpath ../../../
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

load('mat_outputs/vari_per_t60_4.mat')
modelMeans = modelMeans(1:11);
modelSds = modelSds(1:11);
T60s = T60s(1:11);
sem = modelSds./sqrt(num_samples);
ci95 = sem*tinv(.975, num_samples-1);

figure(1)
errorbar(T60s, modelMeans, ci95)
title(sprintf('SSGP Estimator Avg. Error per T60\n (Confidence Intervals: 95%%)'))
xlabel('T60 [s]')
ylabel(sprintf('Mean Error [m^2]\n (Prediction vs. Ground Truth)'))
xlim([0 .8])
ylim([0 max(modelMeans)+.1])

grid
