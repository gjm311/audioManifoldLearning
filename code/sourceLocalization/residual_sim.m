clear
%{
This program is able to detect movement of an array based off of a
misalignment recognition algorithm. The algorithm utilizes MRFs with Fully
Connected latent variables to predict misalignment from sensor
measurements. Here the input is the residual error between an estimated
position by subsets of the overall network and previous estimates. As one would expect, 
if the probability of misalignment is high for all subnets but one, this is 
likely the node that is being moved. More details can be found in
"Misalignment Recognition Using MRFs with Fully connected Latent Variables
for Detecting Localization Failures" by Naoki Akai, Luis Yoichi Morales et
al.
%}

addpath ./functions
addpath ./shortSpeech

% ---- Initialize Parameters ----

%parameters for room
c = 340;
fs = 8000;
roomSize = [6,6,3];    
sourceTrainL = [4,4,1; 2,2,1; 4,2,1; 2,4,1; 3,3,1];
numArrays = 4;
numMics = 2;
nU = 35;
radiusU = max(pdist(sourceTrainL));
nL = size(sourceTrainL,1);
micsPos = [3.025,5,1;2.975,5,1; 5,2.975,1;5,3.025,1; 3.025,1,1;2.975,1,1; 1,2.975,1;1,3.025,1];
ref = roomSize/2;
sourceTrainU = randSourcePos(nU, roomSize, radiusU, ref);
sourceTrain = [sourceTrainL; sourceTrainU];
sourceTest = [3.3, 3, 1];
nD = size(sourceTrain,1);
T60s = .12:.005:.15;
snrs = 0:2:40;
rirLen = 1000;
rtfLen = 500;
% [x,fs_in] = audioread('f0001_us_f0001_00009.wav');
% x = transpose(resample(x,fs,fs_in));
kern_typ = 'gaussian';
numMovePoints = 10;
align_resids = zeros(numArrays,size(T60s,2)*size(snrs,2));
misalign_resids = zeros(numArrays,size(T60s,2)*size(snrs,2));
count = 1;

load('mat_outputs/monoTestSource_biMicCircle_5L50U.mat')
source = zeros(numArrays, size(x,2));
gammaL = inv(sigmaL+rand*10e-3*eye(size(sigmaL)));


pre_p_hat_ts = zeros(numArrays, 3);
for k1 = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, micsPos, RTF_train);
    [~,~,pre_p_hat_ts(k1,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);
end

% ---- Iterate for set of T60s and SNRs to obtain residual errors and latent probabilities ----
for T60 = 1:size(T60s,2)
    for snr = 1:size(snrs,2)
        
        %add noise to each mic array
        for arr = 1:numArrays
           source(arr,:) = awgn(x, snr);
        end
        
        %generate RIRs and estimate RTFs for aligned case (i.e. microphones
        %in same position but with noise and slightly altered T60.
        align_RTF_train = rtfEst(source, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
        align_p_hat_ts = zeros(numArrays, 3);
        for k1 = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, micsPos, align_RTF_train);
            [~,~,align_p_hat_ts(k1,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);
        end
        align_resids(:,count) =  mean(pre_p_hat_ts - align_p_hat_ts,2);
        
        %generate RIRs and estimate RTFs for misaligned case (i.e. mic
        %array now moved and with added noise and slightly altered T60.
        randPos = randi(6,[1,2]);
        randArray = randi(4,1);
        misMicsPos = micsPos;
        if randArray == or(1,3)
            misMicsPos(randArray*2-1:randArray*2,:) = [randPos(1)+.025 randPos(2) 1; randPos(1)-.025 randPos(2) 1];
        else
            misMicsPos(randArray*2-1:randArray*2,:) = [randPos(1) randPos(2)+.025 1; randPos(1) randPos(2)-.025  1]; 
        end
        misalign_RTF_train = rtfEst(source, misMicsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
        misalign_p_hat_ts = zeros(numArrays, 3);
        for k1 = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, misMicsPos, misalign_RTF_train);
            [~,~,misalign_p_hat_ts(k1,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);
        end
        misalign_resids(:,count) =  mean(pre_p_hat_ts - misalign_p_hat_ts,2);
        count = count+1;
        
    end
end
save('mat_outputs/resEsts')