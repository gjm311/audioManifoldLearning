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
load('mat_outputs/movementOptParams')

%simulate different noise levels
snrs = 0:4:40;
iters_per = 10;
noisy_micSignals = zeros(numMics,size(x,2));
for s = 1:size(snrs,1)
    snr = snrs(s);
    for n = 1:numMics
        noisy_signal = awgn(x,snr);
    end
    test_pos = randomSourcePos();    
    for it = 1:iters_per
        
        
        
    end
end
%simulate minimal movement

%simulate multiple arrays moving

%min/max number of labelled estimates needed