clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program looks at the localization error (MSE) of arrays when moved 
varying distances from where they originated (i.e. for training). The
purpose is to see the correspondance between the localization error with
the proabilities of failure of the MRF based movement detector.
%}

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
load('mat_outputs/biMicCircle_5L300U_monoNode')
micRTF_train = RTF_train;
micScales = scales;
micGammaLs = gammaLs;

load('mat_outputs/monoTestSource_biMicCircle_5L300U')
load('mat_results/vari_t60_data.mat')
load('mat_outputs/movementOptParams')
vari = varis_set(I,:);
scales = scales_set(I,:);
[~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
gammaL = inv(sigmaL + diag(ones(1,nL).*vari));

simulate different noise levels
radii = 0:.2:1.2;
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
wavs = dir('./shortSpeech/');

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .1;
lambda = .2;
eMax = .3;
threshes = 0:.02:1;
naive_threshes = .5:.02:1.5;
num_threshes = size(threshes,2);
num_iters = 10;
num_ts = size(T60s,2);

% tp_check = zeros(num_ts,num_threshes);
% fp_check = zeros(num_ts,num_threshes);
% tn_check = zeros(num_ts,num_threshes);
% fn_check = zeros(num_ts,num_threshes);
% 
% subNai_tp_check = zeros(num_ts,num_threshes);
% subNai_fp_check = zeros(num_ts,num_threshes);
% subNai_tn_check = zeros(num_ts,num_threshes);
% subNai_fn_check = zeros(num_ts,num_threshes);

t_str = struct([]);
ts = [1 2 3 4 5 6];

for t = 1:size(ts,2)
    t_curr = ts(t);
    T60 = T60s(t_curr);
    modelMean = modelMeans(t_curr);
    modelSd = modelSds(t_curr);
    RTF_train = reshape(RTF_trains(t_curr,:,:,:), [nD, rtfLen, numArrays]);    
    scales = scales_t(t_curr,:);
    gammaL = reshape(gammaLs(t_curr,:,:), [nL, nL]);

    for thr = 1:num_threshes
        t_str(t,thr).tp_check = mrf_res(1);
        t_str(t,thr).fp_check = mrf_res(2);
        t_str(t,thr).tn_check = mrf_res(3);
        t_str(t,thr).fn_check = mrf_res(4);

        t_str(t,thr).subNai_tp_check = sub_res(1);
        t_str(t,thr).subNai_fp_check = sub_res(2);
        t_str(t,thr).subNai_tn_check = sub_res(3);
        t_str(t,thr).subNai_fn_check = sub_res(4);
        
        thresh = threshes(thr);
        naive_thresh = naive_threshes(thr);
        
        [mrf_res, sub_res] = t60Opt(nD, thresh, naive_thresh, micScales, micGammaLs, micRTF_train, sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
        
        t_str(t,thr).tp_check = mrf_res(1);
        t_str(t,thr).fp_check = mrf_res(2);
        t_str(t,thr).tn_check = mrf_res(3);
        t_str(t,thr).fn_check = mrf_res(4);

        t_str(t,thr).subNai_tp_check = sub_res(1);
        t_str(t,thr).subNai_fp_check = sub_res(2);
        t_str(t,thr).subNai_tn_check = sub_res(3);
        t_str(t,thr).subNai_fn_check = sub_res(4);
        
    end

    save('mat_results/t60results2')    
end

% save('mat_results/thresh_res')

