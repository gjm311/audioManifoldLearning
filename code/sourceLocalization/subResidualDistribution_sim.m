clear
%{
This program estimates the residual error for subnetworks of arrays for a 
series of different T60s and noise levels. This is done for both casess where all
arrays are aligned (i.e. in the same position that was used when the ground 
truth was measured) and one random array is moved to random position in room). In 
residualDistribution_analysis one can see the distribution of the errors 
for the different class types.
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
num_iters = 2700;
align_resids = zeros(num_ts,num_iters);
misalign_resids = zeros(num_ts,num_iters);

wavs = dir('./shortSpeech/');
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];


% ---- Iterate for set of T60s and SNRs to obtain residual errors for aligned and misaligned cases ----
for t = 1:num_ts
    T60 = T60s(t);
    RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
    scales = scales_t(t,:);
    gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
    
   
        
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


        unk_source_pos = randSourcePos(1, roomSize, 3, ref);

        %generate RIRs and estimate RTFs for aligned case (i.e. microphones
        %in same position but with varying noise and T60.

        %---- Initialize subnet estimates of training positions ----
        pre_p_hat_ts = zeros(numArrays, 3); 
        align_p_hat_ts = zeros(numArrays, 3); 
        misalign_p_hat_ts = zeros(numArrays, 3); 
        movingArray = randi(numArrays);
        radius_mic = rand*max_movement;
        [~, micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);

        for k = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
%             [~,~,pre_p_hat_ts(k,:)] = test(unk_x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
%                 sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
            [~,~,align_p_hat_ts(k,:)] = test(unk_x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  

            [subnet, ~, ~] = subNet(k, numArrays, numMics, scales, micsPosNew, RTF_train);
            [~,~,misalign_p_hat_ts(k,:)] = test(unk_x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales); 
        end

        align_resids(t,ni) = mean(mean(((sourceTest-align_p_hat_ts).^2)));
        misalign_resids(t,ni) = mean(mean(((sourceTest-misalign_p_hat_ts).^2)));
    end
    
end
save('mat_outputs/subResEsts_4', 'snrs', 'T60s','align_resids', 'misalign_resids')