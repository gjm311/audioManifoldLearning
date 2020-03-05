%{
This program simulates a room that can be specified in the initialization
parameters with a set of noise sources and microphone nodes. RTFs are
generated (RIRs produced via Emmanuel Habet's RIR generator)and used to
localize sound sources based off a set of learned manifolds (see 
Semi-Supervised Source Localization on Multiple Manifolds With Distributed
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
% ---- Initialize Parameters for training ----
% c = 340;
% fs = 8000;
% roomSize = [6,6,3];    
% height = 2;
% width = 3;
% numRow = 3;
% numCol = 7;
% ref = [roomSize(1:2)./2 1];
% % sourceTrainL = sourceGrid(height, width, numRow, numCol, ref);
% sourceTrainL = [4,4,1; 2,2,1; 4,2,1; 2,4,1; 3,3,1];
% numArrays = 4;
% numMics = 2;
% nU = 100;
% radiusU = max(height,width);
% nL = size(sourceTrainL,1);
% micsPos = [3.025,5,1;2.975,5,1; 5,2.975,1;5,3.025,1; 3.025,1,1;2.975,1,1; 1,2.975,1;1,3.025,1];
% sourceTrainU = randSourcePos(nU, roomSize, radiusU, ref);
% sourceTrain = [sourceTrainL; sourceTrainU];
% nD = size(sourceTrain,1);
% 
% % ---- generate RIRs and estimate RTFs ----
% totalMics = numMics*numArrays;
% rirLen = 1000;
% rtfLen = 500;
% T60 = 0.3;
% x = randn(1,10*fs);
% % [x,fs_in] = audioread('f0001_us_f0001_00209.wav');
% % fs = 32000;
% % x = resample(x,fs_in,fs);
% % x = x';
% disp('Generating the train set');
% 
% %---- Simulate RTFs between nodes ----
% RTF_train = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
% save('mat_trainParams/biMicCircle_5L100U')

% % ---- Simulate RTFs between microphones within each node ----
% RTF_train = zeros(numArrays, nD, rtfLen, numMics);
% for t = 1:numArrays
%     curr_mics = micsPos((t-1)*numMics+1:t*numMics,:); 
%     RTF_train(t,:,:,:) = rtfEst(x, curr_mics, rtfLen, 1, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
% end
% save('mat_trainParams/biMicCircle_5L50U_monoNode')
% 
% ---- load training data (check mat_trainParams for options)----
load('mat_trainParams/biMicCircle_5L100U.mat')

% %Initialize hyper-parameter estimates (gaussian kernel scalar
% %value and noise variance estimate)
% kern_typ = 'gaussian';
% tol = 10e-3;
% alpha = .01;
% max_iters = 100;
% init_scales = 1*ones(1,numArrays);
% var_init = 0;
% [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, init_scales);
% 
% %---- perform grad. descent to get optimal params ----
% gammaL = inv(sigmaL + diag(ones(1,nL)*var_init));
% p_sqL = gammaL*sourceTrainL;
% 
% [costs, ~, ~, varis_set, scales_set] = grad_descent(sourceTrainL, numArrays, RTF_train, sigmaL, init_scales, var_init, max_iters, tol, alpha, kern_typ);
% [~,I] = min(sum(costs));
% % vari = varis_set(I);
% scales = scales_set(I,:);
% vari = .05;

[~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
gammaL = inv(sigmaL + diag(ones(1,nL)*vari));
p_sqL = gammaL*sourceTrainL;

% save('mat_outputs/monoTestSource_biMicCircle_5L100U')
load('mat_outputs/monoTestSource_biMicCircle_5L100U')

%---- with optimal params estimate test position ----
sourceTests = [2.3, 3.3, 1; 3.3, 2.3, 1; 2.3, 2.3, 1; 3.3, 3.3, 1; 4 4 1; 2 2 1];
p_hat_ts = zeros(size(sourceTests));
for i = 1:size(p_hat_ts)
    sourceTest = sourceTests(i,:);
    [~,~, p_hat_ts(i,:)] = test(x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                numMics, sourceTrain, sourceTests(i,:), nL, nU, roomSize, T60, c, fs, kern_typ, scales);
end

%---- plot ----
plotRoom(roomSize, micsPos, sourceTrain, sourceTests, nL, p_hat_ts);

title('Source Localization Based Off Semi-Supervised Approach')
ax = gca;
ax.TitleFontSizeMultiplier  = 2;