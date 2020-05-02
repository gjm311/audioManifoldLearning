
clear
addpath ./RIR-Generator-master
addpath ./functions
% mex -setup c++
% mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program optimizes the potential function of the MRF based on the
empirical prior distributions associated with aligned/misaligned/uncertain
latent classes. Optimization done via iterative proportional fitting (IPF)
method.
%}

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
% load('mat_results/vari_t60_data')
load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('mat_outputs/subResEsts_4')

%---- Set MRF params ----
max_iters = 100;
num_ts = size(T60s,2);
transMats = zeros(num_ts,3,3);
lambdas = .05:.05:.5;
init_vars = .05:.05:.5;
numLams = size(lambdas,2);
numVars = size(init_vars,2);
aucs = zeros(num_ts,numLams,numVars);

wavs = dir('./shortSpeech/');
radii = [0 .65 .85 1.5];
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

for t = 1:num_ts    
    T60 = T60s(t);
    modelMean = modelMeans(t);
    modelSd = modelSds(t);
    RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
    scales = scales_t(t,:);
    gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
    curr_count = 0;
    transMat = ones(3);
    transMat_pre = zeros(3);
    align_resid = align_resids(t,:);
    misalign_resid = misalign_resids(t,:);
    
    while and(norm(transMat_pre-transMat)>=10e-3, curr_count<max_iters)
        curr_count = curr_count + 1;
        transMat_pre = transMat;
                       
        for rad = 1:num_radii
            radius_mic = radii(rad);

            sourceTest = randSourcePos(1, roomSize, radiusU, ref);
            movingArray = randi(numArrays);
            [~, micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
            rand_wav = randi(25);
            try
                file = wavs(rand_wav+2).name;
                [x_tst,fs_in] = audioread(file);
                [numer, denom] = rat(fs/fs_in);
                x_tst = resample(x_tst,numer,denom);
                x_tst = x_tst';  
            catch
                continue
            end    

          %---- Initialize subnet estimates of training positions ----
            sub_p_hat_ts = zeros(numArrays, 3); 
            for k = 1:numArrays
                [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
                [~,~,sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
            end
            
            %Get errors associated for each LONO sub-network
            sub_errors = mean(mean((sourceTest-sub_p_hat_ts).^2));
            
            %Get probability of error according to aligned and
            %misaligned prior distributions
            pEmp = empProbCheck(sub_errors,align_resid,misalign_resid);
            
            %Get probability based on optimal choice of parameters.
            [aucs_curr,pEst] = estProbCheck(radius_mic, lambdas, init_vars, radii,sourceTrain,mic_ref, transMat,wavs,T60,modelMean,modelSd, RTF_train,scales,gammaL,nL, nU,rirLen, rtfLen,c, kern_typ,roomSize,radiusU, ref, numArrays, micsPos, numMics, fs);
            
            %Update via IPF
            transMat = transMat_pre.*(pEmp./(pEst+10e-6));
        end
    end
    aucs(t,:,:) = aucs_curr;
    transMats(t,:,:) = transMat;
    save('./mat_outputs/optTransMatData','aucs','transMats','init_vars','lambdas'); 
end