clear
%{
This program estimates the residual error for a series of different T60s
and noise levels. This is done for both aligned arrays (i.e. in the same 
position that was used when the ground truth was measured) and misaligned
(random array moved to random position in room). In 
residualDistribution_analysis one can see the distribution of the errors 
for the different class types.
%}

addpath ./functions
addpath ./shortSpeech

% ---- Initialize Parameters ----

%parameters for room
load('mat_outputs/monoTestSource_biMicCircle_5L300U.mat')
load('mat_results/vari_t60_data')
t = 1;
RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
scales = scales_t(t,:);
gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
T60s = .15:.15:.9;
snrs = 0:5:40;
align_p_hat_ts = zeros(numArrays, 3);
misalign_p_hat_ts = zeros(numArrays, 3);
unk_p_hat_ts = zeros(numArrays,3);
align_resids = zeros(1,size(T60s,2)*size(snrs,2));
misalign_resids = zeros(1,size(T60s,2)*size(snrs,2));
unk_resids = zeros(numArrays,size(T60s,2)*size(snrs,2)*2);
count = 1;
unk_count = 1;
% source = zeros(numArrays, size(x,2));
% unk_source = zeros(numArrays, size(x,2));
wavs = dir('./shortSpeech/');
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
num_iters = 100;

% pre_p_hat_ts = zeros(numArrays, 3);
% for k1 = 1:numArrays
%     [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, micsPos, RTF_train);
%     [~,~,pre_p_hat_ts(k1,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);
% end

% ---- Iterate for set of T60s and SNRs to obtain residual errors and latent probabilities ----
for t = 1:size(T60s,2)
    T60 = T60s(t);
    
    for s = 1:size(snrs,2)
        snr = snrs(s);
        
        for ni = 1:num_iters
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
            unk_source = zeros(numArrays, size(unk_x,2));
            for arr = 1:numArrays
               unk_source(arr,:) = awgn(unk_x, snr);
            end
            unk_source_pos = randSourcePos(1, roomSize, 3, ref);

            %generate RIRs and estimate RTFs for aligned case (i.e. microphones
            %in same position but with varying noise and T60.
            [~,~,pre_p_hat_t] = test(unk_x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays, numMics, sourceTrain, unk_source_pos, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
            [~,~,align_p_hat_t] = test(unk_source, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays, numMics, sourceTrain, unk_source_pos, nL, nU, roomSize, T60, c, fs, kern_typ, scales);

    %         for k1 = 1:numArrays
    %             [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, micsPos, RTF_train);
    %             [~,~,align_p_hat_ts(k1,:)] = test(source, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);
    %             [~,~,unk_p_hat_ts(k1,:)] = test([source;unk_source], gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, [sourceTest;unk_source_pos], nL, nU, roomSize, T60, c, fs, kern_typ, subscales);
    %         end
    %         align_resids(:,count) =  abs(mean(pre_p_hat_ts - align_p_hat_ts,2));
    %         unk_resids(:,unk_count) =  abs(mean(pre_p_hat_ts - unk_p_hat_ts,2));
    %         unk_count = unk_count+1;

            %Estimate error for misaligned case (i.e. mic
            %array now moved and with added noise and slightly altered T60.
            movingArray = randi(numArrays);
            radius_mic = rand*6;
            [~, micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);

            [~,~,misalign_p_hat_t] = test(unk_source, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays, numMics, sourceTrain, unk_source_pos, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
    %         [~,~,unk_p_hat_t] = test(source, gammaL, RTF_train, misMicPos, rirLen, rtfLen, numArrays, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);

            %         for k1 = 1:numArrays
    %             [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, misMicsPos, RTF_train);
    %             [~,~,misalign_p_hat_ts(k1,:)] = test(source, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);
    %             [~,~,unk_p_hat_ts(k1,:)] = test([source;unk_source], gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, [sourceTest;unk_source_pos], nL, nU, roomSize, T60, c, fs, kern_typ, subscales);        
    %         end
            align_resids(count) =  mean(mean(((pre_p_hat_t-align_p_hat_t).^2)));
            misalign_resids(count) =  mean(mean(((pre_p_hat_t-misalign_p_hat_t).^2)));
    %         unk_resids(unk_count) =  mean(mean(((sourceTest-unk_p_hat_t).^2)));

    %         unk_count = unk_count+1;
            count = count+1;
        end
    end
end
save('mat_outputs/resEsts', 'snrs', 'T60s','align_resids', 'misalign_resids')