clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% load('./mat_results/threshTestResults4')

load('mat_outputs/monoTestSource_biMicCircle_5L300U_2')
load('./paramOpt_results.mat')

radii = 0:.03:.13;
% num_radii = size(radii,2);
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_vars = .05:.1:1.05;
lambdas = .05:.1:.75;
eMax = .3;
threshes = 0:.1:1;
num_threshes = size(threshes,2);
num_iters = 1;
numVaris = size(init_vars,2);
numLams = size(lambdas,2);
num_ts = size(T60s,2);

%--- Interpolate data and plot ROC curve for naive and mrf detectors ---
xq = 1.5:.05:10.5;
for nv = 1:numVaris
    for nl = 1:numLams
        if and(abs(aucs(nl,nv)) > .6, abs(aucs(nl,nv))<.7)
            interp_tprs = sort(interp1(tprs(:,nl,nv),xq));
            interp_fprs = sort(interp1(fprs(:,nl,nv),xq));
             
            
            figure(2)
            if abs(aucs(nl,nv)) == max(max(abs(aucs)))
                mrf_opt = plot(interp_fprs,interp_tprs, '-.p');
            else
                mrf = plot(interp_fprs,interp_tprs, '-.g');
            end
            hold on
            % sub = plot(sub_fpr, sub_tpr, '--b');
            base = plot(threshes,threshes, 'black');
            title(sprintf('ROC Curves: Array Movement Detection\n Curves for varying hyper-paramter choices of the MRF model'))
            xlabel('FPR')
            ylabel('TPR')
            if and(nv == numVaris,nl==numLams)
                legend([mrf,mrf_opt,base], 'ROC for choice of MRF Paramters', 'ROC with Max AUC', 'Baseline', 'Location','southeast')
            end
        end
    end
end

