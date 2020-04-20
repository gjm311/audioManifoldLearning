
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program looks at the localization error (MSE) of arrays when moved 
varying distances from where they originated (i.e. for training). The
purpose is to see the correspondance between the localization error with
the proabilities of failure of the MRF based movement detector.
%}

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
% load('mat_outputs/monoTestSource_biMicCircle_5L300U')

%simulate different noise levels
radii = .05:.2:1.25;
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
num_iters = 100;


load('./mat_results/pFail_res', 'p_fail', 'radii')
p_fail_1 = p_fail;
radii_1 = radii;
load('./mat_results/pFail_res2', 'p_fail', 'radii')
p_fail = [p_fail_1; p_fail];
radii = [radii_1 radii];
figure(1)
bar([0;p_fail])
title(sprintf('Average Probability of Failure for Varying Array Shifts\n (100 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
xlabel(sprintf('Array Shift (m) from Location During Training'))
ylabel(sprintf('Probability of Failure \n (via MRF Detection)'))
mz = {[0 round(radii(1:2:end),2)]};
xticklabels(mz)
% set(gca,'XTickLabel',mz)
ylim([0 1])
