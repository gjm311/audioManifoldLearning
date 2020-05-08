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
load('mat_outputs/resEsts_4')

%---- Set MRF params ----
max_iters = 100;
num_ts = size(T60s,2);
transMats = zeros(num_ts,3,3);

wavs = dir('./shortSpeech/');
radii = [0 1.5 3 4.5 6];
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
    transMat = [1 1 0.05; 1 1 0.05; 1/3 1/3 1/3];
    transMat_pre = zeros(3);
    align_resid = align_resids(t,:);
    misalign_resid = misalign_resids(t,:);
    al_pdParams = fitdist(sort(transpose(align_resid)),'Normal');
    sigma = al_pdParams.sigma;
    mis_pdParams = fitdist(transpose(sort(misalign_resid)),'Exponential');
    lambda = 1/mis_pdParams.mu;

    
    while and(norm(transMat_pre-transMat)>=10e-3, curr_count<max_iters)
        curr_count = curr_count + 1;
        pEmp_al = 0;
        pEmp_mis = 0;
        pEst_al = 0;
        pEst_mis = 0;
                               
        for rad = 1:num_radii
            radius_mic = radii(rad);
            transMat_pre = transMat;

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
            
            %---- estimate test positions before movement ----
            [~,~, pre_p_hat_t] = test(x_tst, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                            numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
            
              %---- estimate test positions after movement ----
            [~,~, p_hat_t] = test(x_tst, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                            numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
            
            %Get errors associated for each LONO sub-network
            sub_error = (mean((pre_p_hat_t-p_hat_t).^2));
            
            %Get probability of error according to aligned and
            %misaligned prior distributions
            [pEmp_al_curr,pEmp_mis_curr] = empProbCheck(sub_error,align_resid,misalign_resid);
            
            %Get probability based on optimal choice of parameters.
            [pEst_al_curr, pEst_mis_curr] = estProbCheck(x_tst, transMat, sigma, lambda, gammaL, numMics, numArrays, micsPosNew, scales, RTF_train,rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);

            pEmp_al = pEmp_al + pEmp_al_curr;
            pEmp_mis = pEmp_mis + pEmp_mis_curr;
            pEst_al = pEst_al + pEst_al_curr;
            pEst_mis = pEst_mis + pEst_mis_curr;
        end
                
        transMat(1,1) = transMat(1,1).*pEmp_al/(pEst_al+10e-6);
        transMat(1,2) = .95-transMat(1,1);
        transMat(2,2) = transMat(2,2).*pEmp_mis/(pEst_mis+10e-6);
        transMat(2,1) = .95-transMat(2,2);

        
    end
    transMats(t,:,:) = transMat;
    save('./mat_outputs/optTransMatData','transMats'); 
end