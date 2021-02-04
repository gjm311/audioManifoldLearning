clear
addpath ./RIR-Generator-master
addpath ./functions
addpath ./stft
addpath ./shortSpeech

%{
This program visualizes the ROC curve with respect to three different detectors:
the MRF based movement detector, as well as naive estimates (mono node
comparison, LONO sub-net comparison).
 
To simulate results, run moveDetect_gt_sim.m
%}


load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')

for r=1:3
    if r==1
        load('./mat_results/gt_indT_results_rotation.mat')
    elseif r==2
        load('./mat_results/gt_nai_results_rotation.mat')
        t_str = mono_t_str;
    else
        load('./mat_results/gt_nai_results_noRotation.mat')       
        t_str = sub_t_str;
    end

    % t_str = mono_t_str;

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

    auc1 = trapz(mean(interp_gt_fprs(1:3,:),1), mean(interp_gt_tprs(1:3,:),1));
    auc2 = trapz(mean(interp_gt_fprs(4:6,:),1), mean(interp_gt_tprs(4:6,:),1));
    auc3 = trapz(mean(interp_gt_fprs(7:9,:),1), mean(interp_gt_tprs(7:9,:),1));
    
    
    figure(r)
    set(gcf,'color','w')
    mrf1 = plot(mean(interp_gt_fprs(1:3,:),1), mean(interp_gt_tprs(1:3,:),1), 'g');
    hold on
    mrf2 = plot(mean(interp_gt_fprs(4:6,:),1), mean(interp_gt_tprs(4:6,:),1), 'b');
    mrf3 = plot(mean(interp_gt_fprs(7:9,:),1), mean(interp_gt_tprs(7:9,:),1),  'r');
    base = plot(threshes,threshes, 'black');
    
    if r==1
        title(sprintf('MRF-Based Detector'))
    elseif r==2
        title(sprintf('Single Node Comp.'))
    else
        title(sprintf('LONO Sub-Net Comp.'))
    end
    
    xlabel('FPR')
    ylabel('TPR')
    legend([mrf1,mrf2,mrf3,base], sprintf(('T60=0.2s, AUC=%g'), round(auc1,2)), sprintf(('T60=0.4s, AUC=%g'), round(auc2,2)), sprintf(('T60=0.6s, AUC=%g'), round(auc3,2)), 'Random Chance', 'Location','southeast')
    grid on
    hold off
end
