clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program simulates the localization error (MSE) of sound sources when arrays 
used in training of the SSGP localizer are moved varying distances from where 
they originated. The purpose is to see the performance of detectors via
the ROC. Here we use the probability of failure as indicated by the MRF based movement detector.
 
To visualize results, run moveDetect_gt_vis.m
%}

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('./mat_results/paramOpt_results_5.mat')
load('mat_outputs/optTransMatData')
num_lambdas = size(lambdas,2);
num_varis = size(init_vars,2);
% load('mat_results/vari_t60_data.mat')
% load('mat_outputs/movementOptParams')

% assign radial shifts of microphone array 
radii = [0 .65 .85 1.5];
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
wavs = dir('./shortSpeech/');

%---- Set MRF params ----
eMax = .3;
threshes = 0:.05:1;
num_threshes = size(threshes,2);
num_iters = 20;
num_ts = size(T60s,2);
t_str = ([]);
gts = [.1 .75 1.45];
num_gts = size(gts,2);

for t = 1:num_ts

    T60 = T60s(t);
    RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
    scales = scales_t(t,:);
    gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
    [lam_pos,var_pos] = find(reshape(aucs(:,:,t),[num_lambdas,num_varis]) == max(max(reshape(aucs(:,:,t),[num_lambdas,num_varis]))));
    lambda = lambdas(lam_pos);
    init_var = init_vars(var_pos);
    transMat = reshape(transMats(t,:,:),[3,3]);

    for g = 1:num_gts
        gt = gts(g);
        
        for thr = 1:num_threshes 
            thresh = threshes(thr);

            mrf_res = gtVary(thresh, sourceTrain, wavs, gammaL, T60, gt, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
            t_str(t,g,thr).tp = mrf_res(1);
            t_str(t,g,thr).fp = mrf_res(2);
            t_str(t,g,thr).tn = mrf_res(3);
            t_str(t,g,thr).fn = mrf_res(4);
            t_str(t,g,thr).tpr = mrf_res(1)/(mrf_res(1)+mrf_res(4)+10e-6);
            t_str(t,g,thr).fpr = mrf_res(2)/(mrf_res(2)+mrf_res(3)+10e-6);

        end
    end
    save('mat_results/gt_indT_results_5', 't_str') 
end   


