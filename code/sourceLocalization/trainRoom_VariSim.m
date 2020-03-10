%{
This program simulates a room that can be specified in the initialization
parameters with a set of noise sources and microphone nodes. RTFs are
generated (RIRs produced via Emmanuel Habet's RIR generator)and used to
=Semi-Supervised Source Localization on Multiple Manifolds With Distributed
Microphones for more details). Gradient descent method used to initialize
hyper-params.
%}
 
% % load desired scenario (scroll past commented section) or uncomment and simulate new environment RTFs 
clear
addpath ./RIR-Generator-master
addpath ./functions
addpath ../../../
load('rtfDatabase.mat')
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');

load('mat_outputs/monoTestSource_biMicCircle_49L400U')

%---- with optimal params estimate test position ----
num_samples = 1000;
diffs = zeros(1,num_samples);
var_conds = zeros(1,num_samples);
p_hat_t = zeros(1,3);


for n = 1:num_samples
    sourceTest = randSourcePos(1, roomSize, radiusU, ref);
    [~,~, p_hat_t] = test(x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
    diffs(n) = norm(sourceTest-p_hat_t);
    RTF_test = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTest, roomSize, T60, rirLen, c, fs);
    sigma_Lt = tstCovEst(nL, nD, numArrays, RTF_train, RTF_test, kern_typ, scales);
    k_t_new = zeros(1,nL);
    for i = 1:nL
        array_kern = 0;
        for j = 1:numArrays
            if numArrays>1
                k_t_new(i) = array_kern + kernel(RTF_train(i,:,j), RTF_test(:,:,j), kern_typ, scales(j));
            else
                k_t_new(i) = array_kern + kernel(RTF_train(i,:), RTF_test(:,:,j), kern_typ, scales(j));
            end
        end
    end
    sigma_Lt = sigma_Lt + (1/numArrays)*k_t_new;
    var_cond(n) = var(sourceTest)-sigma_Lt*gammaL*sigma_Lt';
end
modelSd = std(diffs);
condSd = mean(var_cond);

% load('mat_outputs/monoTestSource_biMicCircle_5L300U.mat')
save('mat_outputs/monoTestSource_biMicCircle_49L400U.mat')

