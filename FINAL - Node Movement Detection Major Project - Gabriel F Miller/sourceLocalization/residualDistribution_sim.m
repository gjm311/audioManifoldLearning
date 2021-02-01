clear
%{
This program estimates the residual error for a series of different T60s
and noise levels. This is done for both aligned arrays (i.e. in the same 
position that was used when the ground truth was measured) and misaligned
(random array moved to random position in room). In 
residualDistribution_vis.m one can see the distribution of the errors 
for the different class types. Distributions generated inform the choices
for the latent state prior distributions of the MRF.
%}

addpath ./functions
addpath ./shortSpeech

% ---- Initialize Parameters ----

%parameters for room
load('mat_outputs/monoTestSource_biMicCircle_5L300U_4.mat')

num_ts = 3;
snrs = 0:5:40;
num_snrs = size(snrs,2);
max_movement = 6;
num_iters = 300;
align_resids = zeros(num_ts,num_snrs*num_iters);
misalign_resids = zeros(num_ts,num_snrs*num_iters);

wavs = dir('./shortSpeech/');
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];


% ---- Iterate for set of T60s and SNRs to obtain residual errors for aligned and misaligned cases ----
for t = 1:num_ts
    T60 = T60s(t);
    RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
    scales = scales_t(t,:);
    gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
    count = 1;
    
    for s = 1:num_snrs
        snr = snrs(s);
        
        for ni = 1:num_iters
            %Choosing random audio file
            try
                rand_wav = randi(25);
                file = wavs(rand_wav+2).name;
                [x_tst,fs_in] = audioread(file);
                [numer, denom] = rat(fs/fs_in);
                x_tst = resample(x_tst,numer,denom);
                unk_x = x_tst';  
            catch
                continue
            end

            %add noise to each mic array
            unk_noisy_x = zeros(numArrays, size(unk_x,2));
            for arr = 1:numArrays
               unk_noisy_x(arr,:) = awgn(unk_x, snr);
            end
            unk_source_pos = randSourcePos(1, roomSize, 3, ref);

            %generate RIRs and estimate RTFs for aligned case (i.e. microphones
            %in same position but with varying noise and T60.
            [~,~,pre_p_hat_t] = test(unk_x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays, numMics, sourceTrain, unk_source_pos, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
            [~,~,align_p_hat_t] = test(unk_noisy_x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays, numMics, sourceTrain, unk_source_pos, nL, nU, roomSize, T60, c, fs, kern_typ, scales);


            %Estimate error for misaligned case (i.e. mic
            %array now moved and with added noise and slightly altered T60.
            movingArray = randi(numArrays);
            radius_mic = 3;
            [~, micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
            [~,~,misalign_p_hat_t] = test(unk_noisy_x, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays, numMics, sourceTrain, unk_source_pos, nL, nU, roomSize, T60, c, fs, kern_typ, scales);

            align_resids(t,count) = mean(mean(((sourceTest-align_p_hat_t).^2)));
            misalign_resids(t,count) = mean(mean(((sourceTest-misalign_p_hat_t).^2)));
            count = count+1;
        end
    end
end
save('mat_outputs/resEsts_4', 'snrs', 'T60s','align_resids', 'misalign_resids')