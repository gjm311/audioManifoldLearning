%{
This program estimates the mean and standard devaition of the source
localization algorithm.
%}
 
% % load desired scenario (scroll past commented section) or uncomment and simulate new environment RTFs 
clear
addpath ./RIR-Generator-master
addpath ./functions
addpath ../../../
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');

load('mat_outputs/monoTestSource_biMicCircle_5L300U_2')

%---- with optimal params estimate test position ----
num_samples = 1000;
diffs = zeros(1,num_samples);
modelMeans = zeros(1,num_ts);
modelSds = zeros(1,num_ts);


for t = 1:num_ts
    T60 = T60s(t);
    RTF_train = reshape(RTF_trains(t,:,:,:),[nD,rtfLen,numArrays]);
    scales = scales_t(t,:);
    gammaL = reshape(gammaLs(t,:,:), [nL,nL]);
    
    for n = 1:num_samples
        sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
        [~,~, p_hat_t] = test(x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                    numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
        diffs(n) = norm(sourceTest-p_hat_t);
    end
    modelMeans(t) = mean(diffs);
    modelSds(t) = std(diffs);
end

save('mat_results/vari_per_t60.mat', 'modelMeans', 'modelSds')