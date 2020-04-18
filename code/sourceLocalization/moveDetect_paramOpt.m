clear
addpath ./RIR-Generator-master
addpath ./functions
% mex -setup c++
% mex RIR-Generator-master/rir_generator.cpp;
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
% load('mat_results/vari_t60_data')
load('mat_outputs/monoTestSource_biMicCircle_5L300U_2')


%simulate different noise levels
radii = 0:.3:1.5;
% num_radii = size(radii,2);
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_vars = .05:.1:1.05;
lambdas = .05:.1:.75;
eMax = .3;
threshes = 0.3:.1:0.7;
num_threshes = size(threshes,2);
num_iters = 1;
numVaris = size(init_vars,2);
numLams = size(lambdas,2);
num_ts = size(T60s,2);

wavs = dir('./shortSpeech/');

tprs = zeros(num_threshes, numLams, numVaris);
fprs = zeros(num_threshes, numLams, numVaris);
aucs = zeros(numLams, numVaris);

for lam = 1:numLams
    lambda = lambdas(lam);

    for v = 1:numVaris
        
        init_var = init_vars(v);
        
        tpr_ch_curr = zeros(1,num_threshes);
        fpr_ch_curr = zeros(1,num_threshes);
            
        for t = 1:num_ts    
            
            T60 = T60s(t);
            modelMean = modelMeans(t);
            modelSd = modelSds(t);
            RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
            scales = scales_t(t,:);
            gammaL = reshape(gammaLs(t,:,:), [nL, nL]);            

            [tpr_out, fpr_out] = paramOpt(sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
            tpr_ch_curr = tpr_ch_curr + tpr_out;
            fpr_ch_curr = fpr_ch_curr + fpr_out;
        end
        tprs(:,lam,v) = tpr_ch_curr./(num_ts);
        fprs(:,lam,v) = fpr_ch_curr./(num_ts);
        aucs(lam,v) = trapz(fprs(:,lam,v),tprs(:,lam,v));

    end
    save('paramOpt_results', 'tprs', 'fprs', 'aucs');   
end

