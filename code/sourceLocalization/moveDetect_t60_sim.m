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
micVaris = varis;
micGammaLs = gammaLs;

load('mat_outputs/monoTestSource_biMicCircle_5L300U')
load('mat_results/vari_t60_data.mat')
% load('mat_outputs/movementOptParams')
% vari = varis_set(I,:);
% scales = scales_set(I,:);
% [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
% gammaL = inv(sigmaL + diag(ones(1,nL).*vari));

%simulate different noise levels
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
naive_threshes = .5:.05:5.5;
num_threshes = size(threshes,2);
num_iters = 10;
num_ts = size(T60s,2);

tp_check = zeros(num_ts,num_threshes);
fp_check = zeros(num_ts,num_threshes);
tn_check = zeros(num_ts,num_threshes);
fn_check = zeros(num_ts,num_threshes);

subNai_tp_check = zeros(num_ts,num_threshes);
subNai_fp_check = zeros(num_ts,num_threshes);
subNai_tn_check = zeros(num_ts,num_threshes);
subNai_fn_check = zeros(num_ts,num_threshes);

nai_tp_check = zeros(num_ts,num_threshes);
nai_fp_check = zeros(num_ts,num_threshes);
nai_tn_check = zeros(num_ts,num_threshes);
nai_fn_check = zeros(num_ts,num_threshes);

mrf_micIDs_corr = zeros(num_ts,num_threshes);
subNai_micIDs_corr = zeros(num_ts,num_threshes);
nai_micIDs_corr = zeros(num_ts,num_threshes);
mrf_micIDs_uncr = zeros(num_ts,num_threshes);
subNai_micIDs_uncr = zeros(num_ts,num_threshes);
nai_micIDs_uncr = zeros(num_ts,num_threshes);


for t = 1:num_ts
    T60 = T60s(t);
    modelMean = modelMeans(t);
    modelSd = modelSds(t);
    RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
    scales = scales_t(t,:);
    gammaL = reshape(gammaLs(t,:,:), [nL, nL]);

    for thr = 1:num_threshes
        thresh = threshes(thr);
        naive_thresh = naive_threshes(thr);
        
        [mrf_res, nai_res, sub_res, mrfID_res, naiID_res, subID_res] = t60Opt(nD, thresh, naive_thresh, micScales, micGammaLs, micRTF_train, sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
        
        tp_check(t,thr) = mrf_res(1);
        fp_check(t,thr) = mrf_res(2);
        tn_check(t,thr) = mrf_res(3);
        fn_check(t,thr) = mrf_res(4);

        subNai_tp_check(t,thr) = sub_res(1);
        subNai_fp_check(t,thr) = sub_res(2);
        subNai_tn_check(t,thr) = sub_res(3);
        subNai_fn_check(t,thr) = sub_res(4);

        nai_tp_check(t,thr) = nai_res(1);
        nai_fp_check(t,thr) = nai_res(2);
        nai_tn_check(t,thr) = nai_res(3);
        nai_fn_check(t,thr) = nai_res(4);

        mrf_micIDs_corr(t,thr) = mrfID_res(1);
        subNai_micIDs_corr(t,thr) = subID_res(1);
        nai_micIDs_corr(t,thr) = naiID_res(1);

        mrf_micIDs_uncr(t,thr) = mrfID_res(2);
        subNai_micIDs_uncr(t,thr) = subID_res(2);
        nai_micIDs_uncr(t,thr) = naiID_res(2);
        
    end

    save('mat_results/t60results')    
    a 
    break
end

save('mat_results/thresh_res')

