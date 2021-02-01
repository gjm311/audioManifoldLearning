clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% load('./mat_results/threshTestResults4')
load('./mat_results/t60results2.mat')
t_str = reshape(t_str(1,:,:),[num_gts, num_threshes]);
for tt = 1:size(t_str


t = 1;
tp_check = reshape(tp_check(t,:), [1,num_threshes]);
fp_check = reshape(fp_check(t,:), [1,num_threshes]);
tn_check = reshape(tn_check(t,:), [1,num_threshes]);
fn_check = reshape(fn_check(t,:), [1,num_threshes]);

subNai_tp_check = reshape(subNai_tp_check(t,:), [1,num_threshes]);
subNai_fp_check = reshape(subNai_fp_check(t,:), [1,num_threshes]);
subNai_tn_check = reshape(subNai_tn_check(t,:), [1,num_threshes]);
subNai_fn_check = reshape(subNai_fn_check(t,:), [1,num_threshes]);

%--- Interpolate data and plot ROC curve for naive and mrf detectors ---
xq = 1.5:.05:51;
mrf_tp_check = interp1(tp_check,xq);
mrf_fp_check = interp1(fp_check,xq);
mrf_tn_check = interp1(tn_check,xq);
mrf_fn_check = interp1(fn_check,xq);
sub_tp_check = interp1(subNai_tp_check,xq);
sub_fp_check = interp1(subNai_fp_check,xq);
sub_tn_check = interp1(subNai_tn_check,xq);
sub_fn_check = interp1(subNai_fn_check,xq);

mrf_tpr = sort(mrf_tp_check./(mrf_tp_check+mrf_fn_check+10e-6));
mrf_fpr = sort(mrf_fp_check./(mrf_fp_check+mrf_tn_check+10e-6));
sub_tpr = sort(sub_tp_check./(sub_tp_check+sub_fn_check+10e-6));
sub_fpr = sort(sub_fp_check./(sub_fp_check+sub_tn_check+10e-6));

figure(1)
mrf = plot(mrf_fpr, mrf_tpr, '-.g');
hold on
sub = plot(sub_fpr, sub_tpr, '--b');
base = plot(threshes,threshes, 'r');
title(sprintf('ROC Curve: Array Movement Detection (T60 = .15s) \n (70 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0 - 1.2m by 10cm increments]'))
xlabel('FPR')
ylabel('TPR')
legend([mrf, sub, base], 'MRF-Based Detector', 'Naive Detector (Single Mic vs Leave One Out SubNet estimate)', 'Baseline', 'Location','southeast')
xlim([0 1.05])
% ylim([0 1.05])

% figure(2)
% heatmap([sum(tp_check) tn_check; fp_check fn_check]);
% 
% figure(2)
% bar(threshes, tp_check)
% title(sprintf('True Positive Movement Detections For Different Probability Thresholds\n (70 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0 - 0.5m by 5cm increments]'))
% xlabel('Probability Threshold')
% ylabel('Frequency of Flags')
% ylim([0 max(tp_check)+5])
% 
% figure(3)
% bar(threshes, fp_check)
% title(sprintf('False Positive Movement Detections For Different Probability Thresholds\n (110 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0 - 0.5m by 5cm increments]'))
% xlabel('Probability Threshold')
% ylabel('Frequency of Flags')
% ylim([0 max(fp_check)+5])
