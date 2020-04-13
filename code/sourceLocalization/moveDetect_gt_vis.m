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

gt_tprs = zeros(num_gts, num_threshes);
gt_fprs = zeros(num_gts, num_threshes);

for tt = 1:num_gts
    for thr = 1:num_threshes
        
    gt_tprs(tt,thr) = t_str(tt,thr).tpr;
    gt_fprs(tt,thr) = t_str(tt,thr).fpr;
    
    end
end

gt = 2;
%--- Interpolate data and plot ROC curve for naive and mrf detectors ---
xq = 1.5:.05:25.5;
gt_tpr = gt_tprs(gt,:);
gt_fpr = gt_fprs(gt,:);

interp_gt_tpr = interp1(gt_tpr,xq);
interp_gt_fpr = interp1(gt_fpr,xq);

% mrf_tp_check = interp1(tp_check,xq);
% mrf_fp_check = interp1(fp_check,xq);
% mrf_tn_check = interp1(tn_check,xq);
% mrf_fn_check = interp1(fn_check,xq);

% mrf_tpr = sort(mrf_tp_check./(mrf_tp_check+mrf_fn_check+10e-6));
% mrf_fpr = sort(mrf_fp_check./(mrf_fp_check+mrf_tn_check+10e-6));


figure(1)
mrf = plot(interp_gt_fpr, interp_gt_tpr, '-.g');
hold on
% sub = plot(sub_fpr, sub_tpr, '--b');
base = plot(threshes,threshes, 'r');
title(sprintf('ROC Curve: Array Movement Detection (T60 = .15s) \n (70 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0 - 1.2m by 10cm increments]'))
xlabel('FPR')
ylabel('TPR')
legend([mrf, base], 'MRF-Based Detector', 'Naive Detector (Single Mic vs Leave One Out SubNet estimate)', 'Baseline', 'Location','southeast')
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
