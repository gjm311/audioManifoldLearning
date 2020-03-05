clear
addpath ./RIR-Generator-master
addpath ./functions
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----

folder = dir('./shortSpeech/');
ridx = randi(numel(folder));
file = folder(ridx).name;