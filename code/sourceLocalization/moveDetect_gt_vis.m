clear
addpath ./RIR-Generator-master
addpath ./functions
% mex -setup c++
% mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% load('./mat_results/threshTestResults4')

load('./mat_results/gt_results_t60_pointEight.mat')
t_str8 = t_str;
load('mat_outputs/monoTestSource_biMicCircle_5L300U_2')
load('./mat_results/gt_results.mat')
t_str(4,:,:) = t_str8;
T60s = [T60s .8];

% simulate different noise levels
radii = [0 .65 .85 1.5];
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
wavs = dir('./shortSpeech/');

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .2;
lambda = .3;
eMax = .3;
threshes = 0:.05:1;
num_threshes = size(threshes,2);
num_iters = 20;
num_ts = size(T60s,2);
gts = [.1 .75 1.45];
num_gts = size(gts,2);

tp = zeros(num_ts,num_gts, num_threshes);
fp = zeros(num_ts,num_gts, num_threshes);
tn = zeros(num_ts,num_gts, num_threshes);
fn = zeros(num_ts,num_gts, num_threshes);
tprs = zeros(num_ts,num_gts, num_threshes);
fprs = zeros(num_ts,num_gts, num_threshes);

for t = 1:num_ts
    for gt = 1:num_gts
        for thr = 1:num_threshes
            tp(t,gt,thr) = t_str(t,gt,thr).tp;
            fp(t,gt,thr) = t_str(t,gt,thr).fp;
            tn(t,gt,thr) = t_str(t,gt,thr).tn;
            fn(t,gt,thr) = t_str(t,gt,thr).fn;
            tprs(t,gt,thr) = t_str(t,gt,thr).tpr;
            fprs(t,gt,thr) = t_str(t,gt,thr).fpr;
        end
    end
end


%--- Interpolate data and plot ROC curve for naive and mrf detectors ---
xq = 1:.001:num_threshes;
interp_gt_tprs = zeros(num_ts*num_gts,size(xq,2));
interp_gt_fprs = zeros(num_ts*num_gts,size(xq,2));

for t = 1:num_ts
    gt_tp = reshape(tp(t,:,:), [num_gts,num_threshes]);
    gt_fp = reshape(fp(t,:,:), [num_gts,num_threshes]);
    gt_tn = reshape(tn(t,:,:), [num_gts,num_threshes]);
    gt_fn = reshape(fn(t,:,:), [num_gts,num_threshes]);
    gt_tprs = reshape(tprs(t,:,:), [num_gts,num_threshes]);
    gt_fprs = reshape(fprs(t,:,:), [num_gts,num_threshes]);

    for gt = 1:num_gts
        interp_gt_tp = interp1(gt_tp(gt,:),xq);
        interp_gt_fp = interp1(gt_fp(gt,:),xq);
        interp_gt_tn = interp1(gt_tn(gt,:),xq);
        interp_gt_fn = interp1(gt_fn(gt,:),xq);

        interp_gt_tprs(gt+(t-1)*num_gts,:) = sort(interp_gt_tp./(interp_gt_tp+interp_gt_fn+10e-6));
        interp_gt_fprs(gt+(t-1)*num_gts,:) = sort(interp_gt_fp./(interp_gt_fp+interp_gt_tn+10e-6));

    %     interp_gt_tprs(gt,:) = sort(interp1(gt_tprs(gt,:),xq));
    %     interp_gt_fprs(gt,:) = sort(interp1(gt_fprs(gt,:),xq));
    end
end

figure(1)
% mrf1 = plot(interp_gt_fprs(1,:), interp_gt_tprs(1,:), 'g');
% hold on
% mrf2 = plot(interp_gt_fprs(2,:), interp_gt_tprs(2,:), 'b');
% mrf3 = plot(interp_gt_fprs(3,:), interp_gt_tprs(3,:), 'r');
% mrf4 = plot(interp_gt_fprs(7,:), interp_gt_tprs(7,:), ':g');
% hold on
% mrf5 = plot(interp_gt_fprs(8,:), interp_gt_tprs(8,:), ':b');
% mrf6 = plot(interp_gt_fprs(9,:), interp_gt_tprs(9,:), ':r');
mrf7 = plot(interp_gt_fprs(10,:), interp_gt_tprs(10,:), '--g');
hold on
mrf8 = plot(interp_gt_fprs(11,:), interp_gt_tprs(11,:), '--b');
mrf9 = plot(interp_gt_fprs(12,:), interp_gt_tprs(12,:), '--r');
base = plot(threshes,threshes, 'black');
title(sprintf('ROC Curve: Array Movement Detection (T60 = .8s) \n Curves for Increasing Ground Truth (GT) Thresholds with Varying Detection Thresholds \n [GTs (meters) are based off shift tolerance]'))
xlabel('FPR')
ylabel('TPR')
% legend([mrf1,mrf2,mrf3,mrf4,mrf5,mrf6,mrf7,mrf8,mrf9,base], 'GT = 0.1m, T60=.15s', 'GT =0.75m, T60=.15s','GT=1.45m, T60=.15s', 'GT = 0.1m, T60=.5s', 'GT =0.75m, T60=.5s','GT=1.45m, T60=.5s', 'GT = 0.1m, T60=.8s', 'GT =0.75m, T60=.8s','GT=1.45m, T60=.8s', 'Baseline', 'Location','southeast')
legend([mrf7,mrf8,mrf9, base],  'GT = 0.1m, T60=.8s', 'GT =0.75m, T60=.8s','GT=1.45m, T60=.8s', 'Baseline', 'Location','southeast')
%     xlim([0 1.05])
% hold off
% ylim([0 1.05])




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


% figure(1)
% mrf1 = plot(reshape(interp_gt_fpr(1,1,:),[1,181]), reshape(interp_gt_tpr(1,1,:),[1,181]), '-.g');
% hold on
% mrf2 = plot(reshape(interp_gt_fpr(1,2,:),[1,181]), reshape(interp_gt_tpr(1,2,:),[1,181]), '-.r');
% mrf3 = plot(reshape(interp_gt_fpr(1,3,:),[1,181]), reshape(interp_gt_tpr(1,3,:),[1,181]), '-.b');
% % sub = plot(sub_fpr, sub_tpr, '--b');
% mrf4 = plot(reshape(interp_gt_fpr(2,1,:),[1,181]), reshape(interp_gt_tpr(2,1,:),[1,181]), '-*g');
% mrf5 = plot(reshape(interp_gt_fpr(2,2,:),[1,181]), reshape(interp_gt_tpr(2,2,:),[1,181]), '-*r');
% mrf6 = plot(reshape(interp_gt_fpr(2,3,:),[1,181]), reshape(interp_gt_tpr(2,3,:),[1,181]), '-*b');
% % sub = plot(sub_fpr, sub_tpr, '--b');
% base = plot(threshes,threshes, 'black');
% title(sprintf('ROC Curve: Array Movement Detection (T60 = .15s, .5s) \n Curves for Increasing Ground Truth (GT) Thresholds with Varying Detection Thresholds \n [GT in meters for diffrence in positional estimate before/after movement]'))
% xlabel('FPR')
% ylabel('TPR')
% legend([mrf1,mrf2,mrf3,mrf4,mrf5,mrf6,base], 'GT = 0m, T60=.15s', 'GT =0.4m, T60=.15s','GT=.8m, T60=.15s','GT = 0m, T60=.5s', 'GT =0.4m, T60=.5s','GT=.8m, T60=.5s', 'Baseline', 'Location','southeast')
%     xlim([0 1.05])
% hold off
% ylim([0 1.05])

% 
% figure(2)
% heatmap([sum(tp_check) tn_check; fp_check fn_check]);
% tp_check = zeros(1,num_threshes);
% fp_check = zeros(1,num_threshes);
% tn_check = zeros(1,num_threshes);
% fn_check = zeros(1,num_threshes);
% 
% t = 1;
% gt = 1;
% for th = 1:num_threshes
%     tp_check(th) = t_str(t,gt,th).tp; 
%     fp_check(th) = t_str(t,gt,th).fp; 
%     tn_check(th) = t_str(t,gt,th).tn; 
%     fn_check(th) = t_str(t,gt,th).fn;     
% end
% 
% figure(2)
% bar(threshes, fn_check)
% title(sprintf('False Negative Movement Detections For Different Probability Thresholds\n (T60: .15s, Ground Truth: .1m)'))
% xlabel('Probability Threshold')
% ylabel('Frequency of Flags')
% ylim([0 80])
% 
% figure(3)
% bar(threshes, fn_check)
% title(sprintf('False Positive Movement Detections For Different Probability Thresholds\n (110 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0 - 0.5m by 5cm increments]'))
% xlabel('Probability Threshold')
% ylabel('Frequency of Flags')
% ylim([0 max(fn_check)+5])
