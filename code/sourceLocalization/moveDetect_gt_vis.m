clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% load('./mat_results/threshTestResults4')

load('mat_outputs/monoTestSource_biMicCircle_5L300U_2')
load('./mat_results/gt_results.mat')
% load('mat_results/vari_t60_data.mat')
% load('mat_outputs/movementOptParams')

% simulate different noise levels
radii = .2:.3:.8;
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
wavs = dir('./shortSpeech/');

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .2;
lambda = .3;
eMax = .3;
threshes = 0:.1:1;
num_threshes = size(threshes,2);
num_iters = 1;
num_ts = size(T60s,2);
gts = 0:.15:.6;
num_gts = size(gts,2);
% t_str = reshape(t_str(1,:,:),[5, 11]);

gt_tp = zeros(num_radii, num_threshes);
gt_fp = zeros(num_radii, num_threshes);
gt_tn = zeros(num_radii, num_threshes);
gt_fn = zeros(num_radii, num_threshes);
gt_tprs = zeros(num_radii, num_threshes);
gt_fprs = zeros(num_radii, num_threshes);
interp_gt_tpr = zeros(2,3,181);
interp_gt_fpr = zeros(2,3,181);

 for t = 1:2
     for gt = 1:3
        for tt = 1:num_radii
            for thr = 1:num_threshes
                gt_tp(tt,thr) = t_str(t,tt,thr).tp;
                gt_fp(tt,thr) = t_str(t,tt,thr).fp;
                gt_tn(tt,thr) = t_str(t,tt,thr).tn;
                gt_fn(tt,thr) = t_str(t,tt,thr).fn;
                gt_tprs(tt,thr) = t_str(t,tt,thr).tpr;
                gt_fprs(tt,thr) = t_str(t,tt,thr).fpr;

            end
        end
% 
%         gt = 2;
        %--- Interpolate data and plot ROC curve for naive and mrf detectors ---
        xq = 1.5:.05:10.5;
        interp_gt_tp = interp1(gt_tp(gt,:),xq);
        interp_gt_fp = interp1(gt_fp(gt,:),xq);
        interp_gt_tn = interp1(gt_tn(gt,:),xq);
        interp_gt_fn = interp1(gt_fn(gt,:),xq);

        interp_gt_tpr(t,gt,:) = sort(interp_gt_tp./(interp_gt_tp+interp_gt_fn+10e-6));
        interp_gt_fpr(t,gt,:) = sort(interp_gt_fp./(interp_gt_fp+interp_gt_tn+10e-6));
     end
 end
% gt_tpr = sort(gt_tprs(gt,:));
% gt_fpr = sort(gt_fprs(gt,:));
 
% interp_gt_tpr = interp1(gt_tpr,xq);
% interp_gt_fpr = interp1(gt_fpr,xq);

% mrf_tp_check = interp1(tp_check,xq);
% mrf_fp_check = interp1(fp_check,xq);
% mrf_tn_check = interp1(tn_check,xq);
% mrf_fn_check = interp1(fn_check,xq);

% mrf_tpr = sort(mrf_tp_check./(mrf_tp_check+mrf_fn_check+10e-6));
% mrf_fpr = sort(mrf_fp_check./(mrf_fp_check+mrf_tn_check+10e-6));


figure(1)
mrf1 = plot(reshape(interp_gt_fpr(1,1,:),[1,181]), reshape(interp_gt_tpr(1,1,:),[1,181]), '-.g');
hold on
mrf2 = plot(reshape(interp_gt_fpr(1,2,:),[1,181]), reshape(interp_gt_tpr(1,2,:),[1,181]), '-.r');
mrf3 = plot(reshape(interp_gt_fpr(1,3,:),[1,181]), reshape(interp_gt_tpr(1,3,:),[1,181]), '-.b');
% sub = plot(sub_fpr, sub_tpr, '--b');
mrf4 = plot(reshape(interp_gt_fpr(2,1,:),[1,181]), reshape(interp_gt_tpr(2,1,:),[1,181]), '-*g');
mrf5 = plot(reshape(interp_gt_fpr(2,2,:),[1,181]), reshape(interp_gt_tpr(2,2,:),[1,181]), '-*r');
mrf6 = plot(reshape(interp_gt_fpr(2,3,:),[1,181]), reshape(interp_gt_tpr(2,3,:),[1,181]), '-*b');
% sub = plot(sub_fpr, sub_tpr, '--b');
base = plot(threshes,threshes, 'black');
title(sprintf('ROC Curve: Array Movement Detection (T60 = .15s, .5s) \n Curves for Increasing Ground Truth (GT) Thresholds with Varying Detection Thresholds \n [GT in meters for diffrence in positional estimate before/after movement]'))
xlabel('FPR')
ylabel('TPR')
legend([mrf1,mrf2,mrf3,mrf4,mrf5,mrf6,base], 'GT = 0m, T60=.15s', 'GT =0.4m, T60=.15s','GT=.8m, T60=.15s','GT = 0m, T60=.5s', 'GT =0.4m, T60=.5s','GT=.8m, T60=.5s', 'Baseline', 'Location','southeast')
%     xlim([0 1.05])
% hold off
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
