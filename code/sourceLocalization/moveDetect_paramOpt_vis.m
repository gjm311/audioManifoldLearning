clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% load('./mat_results/threshTestResults4')

load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('./mat_results/paramOpt_results_4.mat')

radii = [0 .65 .85 1.5];
% num_radii = size(radii,2);
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_vars = .05:.1:.55;
lambdas = .05:.1:.55;
eMax = .3;
threshes = 0:.25:1;
num_threshes = size(threshes,2);
num_iters = 5;
numVaris = size(init_vars,2);
numLams = size(lambdas,2);
num_ts = size(T60s,2);




for t=1:num_ts
    figure(t)
%     h = heatmap(reshape(aucs(:,:,t),[numLams,numVaris]));
    h = heatmap(mean(aucs,3));
%     title(sprintf('AUCs for Varying MRF Hyper-Parameters \nT-60 = %s',num2str(round(T60s(t),2))))
    title(sprintf('AUCs for Varying MRF Hyper-Parameters \n Averaged for all T-60s'))
    ylabel('\lambda')
    xlabel('\sigma^2')
    % ax = gca;
    h.XData = round(lambdas,2);
    h.YData = round(init_vars,2);
    ax = gca;
end

%--- Interpolate data and plot ROC curve for naive and mrf detectors ---
% xq = 1.5:.05:10.5;
% for nv = 1:numVaris
%     for nl = 1:numLams
% 
%         interp_tprs = sort(tprs(:,nl,nv));
%         interp_fprs = sort(fprs(:,nl,nv));
% 
%         figure(2)
%         if aucs(nl,nv) == max(max(aucs))
%             mrf_opt = plot(interp_fprs,interp_tprs, 'p');
%         else
%             mrf = plot(interp_fprs,interp_tprs, ':g');
%         end
%         hold on
%         % sub = plot(sub_fpr, sub_tpr, '--b');
%         base = plot(threshes,threshes, 'black');
%         title(sprintf('ROC Curves: Array Movement Detection\n Curves for varying hyper-paramter choices of the MRF model'))
%         xlabel('FPR')
%         ylabel('TPR')
%         if and(nv == numVaris,nl==numLams)
%             legend([mrf,mrf_opt,base], 'ROC for choice of MRF Paramters', 'ROC with Max AUC', 'Baseline', 'Location','southeast')
%         end
%        
%     end
% end
% 
