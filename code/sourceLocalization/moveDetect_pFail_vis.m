clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program looks at the localization error (MSE) and posterior probabilities
output from the MRF model when arrays are moved varying distances from where 
they were during training). Results show the benefit of using the proabilities 
of failure of the MRF based movement detector rather than localization
error.
%}


%%---- Uncomment for visualizing posterior probability results ----
% load('./mat_results/pFail_res', 'p_fail', 'radii')
% p_fail_1 = p_fail;
% radii_1 = radii;
% load('./mat_results/pFail_res2', 'p_fail', 'radii')
% p_fail = [p_fail_1; p_fail];
% radii = [radii_1 radii];
% figure(1)
% bar([0;p_fail])
% title(sprintf('Average Probability of Failure for Varying Array Shifts\n (100 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
% xlabel(sprintf('Array Shift (m) from Location During Training'))
% ylabel(sprintf('Probability of Failure \n (via MRF Detection)'))
% mz = {[0 round(radii(1:2:end),2)]};
% xticklabels(mz)
% % set(gca,'XTickLabel',mz)
% ylim([0 1])


%---- Uncomment for localization results for increasing shifts.
load('./mat_results/loclError_res', 'localErrors', 'radii')
localError = mean(localErrors,2);
figure(1)
bar([0;localError])
title(sprintf('Average Probability of Failure for Varying Array Shifts\n (100 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
xlabel(sprintf('Array Shift (m) from Location During Training'))
ylabel(sprintf('Estimation Error (m) After Shift'))
mz = {[0 round(radii(1:2:end),2)]};
xticklabels(mz)
% set(gca,'XTickLabel',mz)
ylim([0 1])
