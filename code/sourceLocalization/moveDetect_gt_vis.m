clear
addpath ./RIR-Generator-master
addpath ./functions
% mex -setup c++
% mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% load('./mat_results/threshTestResults4')

% load('./mat_results/gt_results_t60_pointEight.mat')
% t_str8 = t_str;
load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('./mat_results/gt_monoNai_results_4.mat')
% t_str(4,:,:) = t_str8;
% T60s = [T60s .8];

t_str = sub_t_str;

% simulate different noise levels
radii = [0 .65 .85 1.5];
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
wavs = dir('./shortSpeech/');

%---- Set MRF params ----
threshes = 0:.05:1;
num_threshes = size(t_str,3);
num_iters = 20;
num_ts = size(T60s,2);
gts = [.1 .75 1.45];
num_gts = size(gts,2);

tp = zeros(num_ts,num_gts,num_threshes);
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

%         interp_gt_tprs(gt,:) = sort(interp1(gt_tprs(gt,:),xq));
%         interp_gt_fprs(gt,:) = sort(interp1(gt_fprs(gt,:),xq));
    end
end

figure(1)
mrf1 = plot(mean(interp_gt_fprs(1:3,:),1), mean(interp_gt_tprs(1:3,:),1), 'g');
hold on
mrf2 = plot(mean(interp_gt_fprs(4:6,:),1), mean(interp_gt_tprs(4:6,:),1), 'b');
mrf3 = plot(mean(interp_gt_fprs(7:9,:),1), mean(interp_gt_tprs(7:9,:),1),  'r');
base = plot(threshes,threshes, 'black');
title(sprintf('MRF based Movement Detection ROC Curves\n [MRF parameters individually tuned per T60]'))
% title(sprintf('MRF based Movement Detection ROC Curves\n [MRF parameters chosen based on AUCs averaged per T60]'))
% title(sprintf('MRF based Movement Detection ROC Curves\n [MRF parameters chosen based off empirically derived distributions]'))
xlabel('FPR')
ylabel('TPR')
legend([mrf1,mrf2,mrf3,base], 'T60=.2s', 'T60=.4s', 'T60=.6s', 'Baseline', 'Location','southeast')

auc1 = trapz(mean(interp_gt_fprs(1:3,:),1), mean(interp_gt_tprs(1:3,:),1))
auc2 = trapz(mean(interp_gt_fprs(4:6,:),1), mean(interp_gt_tprs(4:6,:),1))
auc3 = trapz(mean(interp_gt_fprs(7:9,:),1), mean(interp_gt_tprs(7:9,:),1))

% figure(1)
% mrf1 = plot(interp_gt_fprs(1:3,:), interp_gt_tprs(1:3,:), 'g');
% hold on
% mrf2 = plot(interp_gt_fprs(2,:), interp_gt_tprs(2,:), 'b');
% mrf3 = plot(interp_gt_fprs(3,:), interp_gt_tprs(3,:), 'r');
% mrf4 = plot(interp_gt_fprs(4,:), interp_gt_tprs(4,:), ':g');
% % hold on
% mrf5 = plot(interp_gt_fprs(5,:), interp_gt_tprs(5,:), ':b');
% mrf6 = plot(interp_gt_fprs(6,:), interp_gt_tprs(6,:), ':r');
% mrf7 = plot(interp_gt_fprs(7,:), interp_gt_tprs(7,:), '--g');
% % hold on
% mrf8 = plot(interp_gt_fprs(8,:), interp_gt_tprs(8,:), '--b');
% mrf9 = plot(interp_gt_fprs(9,:), interp_gt_tprs(9,:), '--r');
% base = plot(threshes,threshes, 'black');
% title(sprintf('ROC Curve: Array Movement Detection (T60 = [0.2, 0.4, 0.6s]) \n Curves for Increasing Ground Truth (GT) Thresholds with Varying Detection Thresholds \n [GTs (meters) are based off shift tolerance]'))
% xlabel('FPR')
% ylabel('TPR')
% legend([mrf1,mrf2,mrf3,mrf4,mrf5,mrf6,mrf7,mrf8,mrf9,base], 'GT = 0.1m, T60=.2s', 'GT =0.75m, T60=.2s','GT=1.45m, T60=.2s', 'GT = 0.1m, T60=.4s', 'GT =0.75m, T60=.4s','GT=1.45m, T60=.4s', 'GT = 0.1m, T60=.6s', 'GT =0.75m, T60=.6s','GT=1.45m, T60=.6s', 'Baseline', 'Location','southeast')

% figure(2)
% mrf1 = plot(interp_gt_fprs(1,:), interp_gt_tprs(1,:), 'g');
% hold on
% mrf2 = plot(interp_gt_fprs(2,:), interp_gt_tprs(2,:), 'b');
% mrf3 = plot(interp_gt_fprs(3,:), interp_gt_tprs(3,:), 'r');
% base = plot(threshes,threshes, 'black');
% title(sprintf('ROC Curve: Array Movement Detection (T60 = 0.2s) \n Curves for Increasing Ground Truth (GT) Thresholds with Varying Detection Thresholds \n [GTs (meters) are based off shift tolerance]'))
% xlabel('FPR')
% ylabel('TPR')
% legend([mrf1,mrf2,mrf3,base], 'GT = 0.1m, T60=.2s', 'GT =0.75m, T60=.2s','GT=1.45m, T60=.2s', 'Baseline', 'Location','southeast')
% 
% figure(3)
% mrf4 = plot(interp_gt_fprs(4,:), interp_gt_tprs(4,:), ':g');
% hold on
% mrf5 = plot(interp_gt_fprs(5,:), interp_gt_tprs(5,:), ':b');
% mrf6 = plot(interp_gt_fprs(6,:), interp_gt_tprs(6,:), ':r');
% base = plot(threshes,threshes, 'black');
% title(sprintf('ROC Curve: Array Movement Detection (T60 = 0.4s) \n Curves for Increasing Ground Truth (GT) Thresholds with Varying Detection Thresholds \n [GTs (meters) are based off shift tolerance]'))
% xlabel('FPR')
% ylabel('TPR')
% legend([mrf4,mrf5,mrf6,base], 'GT = 0.1m, T60=.4s', 'GT =0.75m, T60=.4s','GT=1.45m, T60=.4s', 'Baseline', 'Location','southeast')
% 
% figure(4)
% mrf7 = plot(interp_gt_fprs(7,:), interp_gt_tprs(7,:), '--g');
% hold on
% mrf8 = plot(interp_gt_fprs(8,:), interp_gt_tprs(8,:), '--b');
% mrf9 = plot(interp_gt_fprs(9,:), interp_gt_tprs(9,:), '--r');
% base = plot(threshes,threshes, 'black');
% title(sprintf('ROC Curve: Array Movement Detection (T60 = 0.6s) \n Curves for Increasing Ground Truth (GT) Thresholds with Varying Detection Thresholds \n [GTs (meters) are based off shift tolerance]'))
% xlabel('FPR')
% ylabel('TPR')
% legend([mrf7,mrf8,mrf9, base],  'GT = 0.1m, T60=.6s', 'GT =0.75m, T60=.6s','GT=1.45m, T60=.6s', 'Baseline', 'Location','southeast')
