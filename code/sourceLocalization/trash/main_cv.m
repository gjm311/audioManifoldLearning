% ---- This program...

% % load desired scenario or run new
clear
addpath ./RIR-Generator-master
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% 
% c = 340;
% fs = 8000;
% roomSize = [6,6,3]; 
% height = 2;
% width = 3;
% numRow = 3;
% numCol = 7;
% ref = roomSize/2;
% sourceTrainL = sourceGrid(height, width, numRow, numCol, ref);
% % sourceTrainL = [3.1,3,1;2.9,3,1;];
% 
% sourceTest = [3, 3, 1];
% numArrays = 4;
% numMics = 2;
% nU = 50;
% radiusU = max(height,width);
% nL = size(sourceTrainL,1);
% micsPos = [3.1,5,1;2.9,5,1; 5,2.9,1;5,3.1,1; 3.1,1,1;2.9,1,1; 1,2.9,1;1,3.1,1];
% sourceTrainU = randSourcePos(nU,roomSize, radiusU, ref);
% sourceTrain = [sourceTrainL; sourceTrainU];
% nD = size(sourceTrain,1);
% 
% % ---- generate RIRs and estimate RTFs ----
% totalMics = numMics*numArrays;
% rirLen = 1000;
% rtfLen = 500;
% T60 = 0.15;
% train_sp = randn(size(sourceTrain,1), 10*fs);
% x = randn(1,10*fs);
% % [x,fs_in] = audioread('f0001_us_f0001_00009.wav');
% % x = transpose(resample(x,fs,fs_in));
% disp('Generating the train set');
% RTF_train = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
% 
% 
% save('mat_trainParams/singleTestSource_biMicCircle_gridL')
% 
% % Load train manifold params based on labelled and unlabelled training data
load('mat_trainParams/singleNoiseSource_biMicCircle_100U')

%---- generate range of gaussian scalar values and dBWs to x-val over ----
scales_lin = 0.01:.1:1;
numScales = size(scales_lin,2);
scales_set = ones(numScales,numArrays);
for i = 1:numScales
   scales_set(i,:) =  scales_set(i,:)*scales_lin(i);
end

dBWs = -20:5:35;
numDBWs = size(dBWs,2);
scores = zeros(numScales, numDBWs);
sourceTrPos = zeros(numScales*numDBWs,nL,3);

for i = 1:numScales
    for j = 1:numDBWs
        scales = scales_set(i,:);
        dBW = dBWs(j);
        
        %---- calculate covariance for labelled nodes ----
        kern_typ = 'gaussian';
        sigmaL = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
        
        %---- get labelled position values ----
        wn = wgn(1,nL,dBW);
        vari = var(wn);
        gammaL = inv(sigmaL + diag(ones(1,nL)*vari));
        p_sqL = gammaL*sourceTrainL;
        
        %---- evaluate norm difference between actual train and test
        %sources and estimates.
        score = crossentropy(reshape(p_sqL,[1,3*nL]), reshape(sourceTrainL, [1,3*nL]));
        scores(i,j) = score;
        sourceTrPos(numDBWs*(i-1)+j,:,:) = p_sqL;
    end
end

[~,I] = min(scores);
opt_p_sqL = sourceTrPos(numDBWs*(I(1)-1)+I(2),:);
opt_scales = scales_set(I(1),:);
opt_dBW = dBWs(I(2));
wn = wgn(1,nL,opt_dBW);
vari = var(wn);

%---- estimate test position ----
[p_sqL_new, p_hat_t] = test(x, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
    numMics, sourceTrain, sourceTest, nL, nU, roomSize, vari, T60, c, fs, kern_typ, opt_scales);

% 

%---- plot ----
plotRoom(roomSize, micsPos, sourceTrain, sourceTest, nL, p_sqL_new, p_hat_t);
title('training grid')



