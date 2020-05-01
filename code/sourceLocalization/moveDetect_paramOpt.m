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
load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')


%set radii shifts of microphone array
radii = [0 .65 .85 1.5];
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
selfs = .6:.05:.9;
opps = .4:.05:.1;
transMats = zeros(10,3,3);
init_vars = .05:.1:.55;
lambdas = .05:.1:.55;
eMax = .3;
threshes = 0:.25:1;
num_threshes = size(threshes,2);
num_iters = 5;
numVaris = size(init_vars,2);
numLams = size(lambdas,2);
num_ts = size(T60s,2);

wavs = dir('./shortSpeech/');

tprs = zeros(numTrans,numLams, numVaris,num_ts, num_threshes);
fprs = zeros(numLams, numVaris, num_ts, num_threshes);
aucs = zeros(numLams, numVaris, num_ts); 

for tr = 1:numTrans
    transmats(tr,3,:) = [1/3 1/3 1/3];
    
    
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
                tprs(lam,v,t,:) = tpr_out;
                fprs(lam,v,t,:) = fpr_out;
                aucs(lam,v,t) = trapz(flip(fpr_out),flip(tpr_out));
    %             tpr_ch_curr = tpr_ch_curr + tpr_out;
    %             fpr_ch_curr = fpr_ch_curr + fpr_out;
            end
    %         tprs(:,lam,v) = tpr_ch_curr./(num_ts);
    %         fprs(:,lam,v) = fpr_ch_curr./(num_ts);
    %         aucs(lam,v) = trapz(fprs(:,lam,v),tprs(:,lam,v));

        end
        save('./mat_results/paramOpt_results_4', 'tprs', 'fprs', 'aucs', 'init_vars', 'lambdas', 'transMats');   
    end
end
