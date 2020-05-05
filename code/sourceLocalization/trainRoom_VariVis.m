%{
This program simulates a room that can be specified in the initialization
parameters with a set of noise sources and microphone nodes. RTFs are
generated (RIRs produced via Emmanuel Habet's RIR generator)and used to
=Semi-Supervised Source Localization on Multiple Manifolds With Distributed
Microphones for more details). Gradient descent method used to initialize
hyper-params.
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
