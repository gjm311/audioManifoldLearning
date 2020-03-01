clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_biMicCircle_5L50U.mat')


%simulate different noise levels
snrs = 0:40;

randomSearch()



%simulate minimal movement

%simulate multiple arrays moving

%min/max number of labelled estimates needed