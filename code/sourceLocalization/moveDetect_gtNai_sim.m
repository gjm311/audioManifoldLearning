clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program simulates tprs and fprs for the naive case (i.e. estimation of
movement determined by comparing single node estimates and sub networks
with the node that was left out.
%}

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
% load('mat_outputs/biMicCircle_5L300U_monoNode_2')
% micRTF_trains = RTF_train;
% micScales = scales;
% micGammaLs = gammaLs;

load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('mat_outputs/biMicCircle_5L300U_monoNode_4')
% load('mat_results/vari_t60_data.mat')
% load('mat_outputs/movementOptParams')

% assign radial shifts of microphone array 
radii = [0 .65 .85 1.5];
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
wavs = dir('./shortSpeech/');

%---- Set MRF params ----
mono_threshes = [0 0.5:.25:2.5 100];
sub_threshes = [0:1.25:11.25 10000];
num_threshes = size(sub_threshes,2);
num_iters = 20;
num_ts = size(T60s,2);
mono_t_str = ([]);
sub_t_str = ([]);
gts = [.1 .75 1.45];
num_gts = size(gts,2);

for t = 1:num_ts

    T60 = T60s(t);
    RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
    scales = scales_t(t,:);
    gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
    
    micRTF_train = reshape(micRTF_trains(t,:,:,:), [numArrays, nD, rtfLen, numMics]);
    micScale = reshape(micScales(t,:,:), [numArrays,numMics]);
    micGammaL = reshape(micGammaLs(t,:,:,:), [numArrays,nL,nL]);

    for g = 1:num_gts
        gt = gts(g);
        
        for thr = 1:num_threshes 
            mono_thresh = mono_threshes(thr);
            sub_thresh = sub_threshes(thr);

            [mono_res,sub_res] = gtNaiVary(mono_thresh, sub_thresh, sourceTrain, wavs, gammaL, T60, gt, micRTF_train, micScale, micGammaL, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
            mono_t_str(t,g,thr).tp = mono_res(1);
            mono_t_str(t,g,thr).fp = mono_res(2);
            mono_t_str(t,g,thr).tn = mono_res(3);
            mono_t_str(t,g,thr).fn = mono_res(4);
            mono_t_str(t,g,thr).tpr = mono_res(1)/(mono_res(1)+mono_res(4)+10e-6);
            mono_t_str(t,g,thr).fpr = mono_res(2)/(mono_res(2)+mono_res(3)+10e-6);
            
            sub_t_str(t,g,thr).tp = sub_res(1);
            sub_t_str(t,g,thr).fp = sub_res(2);
            sub_t_str(t,g,thr).tn = sub_res(3);
            sub_t_str(t,g,thr).fn = sub_res(4);
            sub_t_str(t,g,thr).tpr = sub_res(1)/(sub_res(1)+sub_res(4)+10e-6);
            sub_t_str(t,g,thr).fpr = sub_res(2)/(sub_res(2)+sub_res(3)+10e-6);

        end
    end
    save('mat_results/gt_nai_results_4', 'mono_t_str', 'sub_t_str') 
end   


