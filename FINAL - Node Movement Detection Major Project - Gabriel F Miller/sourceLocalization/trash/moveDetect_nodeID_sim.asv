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
micRTF_trains = RTF_train;
micScales = scales;
micGammaLs = gammaLs;
load('mat_outputs/monoTestSource_biMicCircle_5L300U')
load('mat_results/vari_t60_data.mat')
% load('mat_outputs/movementOptParams')

% simulate different noise levels
radii = .8;
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
wavs = dir('./shortSpeech/');

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .1;
lambda = .2;
eMax = .3;
threshes = 0:.1:1;
num_threshes = size(threshes,2);
num_iters = 100;
noises = 10:5:30;
num_ts = size(T60s,2);
num_ns = size(noises,2);
t_str = ([]);

for t = 1:num_ts
    T60 = T60s(t);
    for n = 1:num_ns
        noise = noises(n);
        RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
        scales = scales_t(t,:);
        gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
        micRTF_train = reshape(micRTF_trains(t,:,:,:), [numArrays, nD, rtfLen, numMics]);
        micScale = reshape(micScales(t,:,:), [numArrays,numMics]);
        micGammaL = reshape(micGammaLs(t,:,:,:), [numArrays,nL,nL]);

        [mrf_res,nai_res,sub_res] = nodeID(noise, micScale, micGammaL, micRTF_train, sourceTrain, wavs, gammaL, T60, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
   
        t_str(t,n).mrf_acc = mrf_res(1)/(mrf_res(1)+mrf_res(2)+10e-6);
        t_str(t,n).nai_acc = nai_res(1)/(nai_res(1)+nai_res(2)+10e-6);
        t_str(t,n).sub_acc = sub_res(1)/(sub_res(1)+sub_res(2)+10e-6);
    end
end
save('mat_results/nodeID_results', 't_str') 
% save('mat_results/thresh_res')



