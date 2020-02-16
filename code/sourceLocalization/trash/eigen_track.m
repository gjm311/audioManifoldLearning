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
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');

%---- load rtf sample data ----
load('mat_trainParams/biMicCircle_5L50U.mat')
load('mat_trainParams/rtfDatabase.mat')

devs = zeros(nL,size(rtfDatabase,1));

for i = 1:size(rtfDatabase,1)
    %Load RTF samples and estimate covariance mtx.
    RTF_train = rtfDatabase(i).b;
    micsPos = [micsPos(1:4,:); rtfDatabase(i).a; micsPos(7:8,:)]; 
    kern_typ = 'gaussian';
    tol = 10e-3;
    alpha = 10e-3;
    max_iters = 100;
    init_scales = 1*ones(1,numArrays);
    var_init = 0;
    sigmaL = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, init_scales);
    [vecs,vals] = eigs(sigmaL);
    devs(:,i) = vecs(:,nL);
    plotRoom(roomSize, micsPos, sourceTrain,[],nL,[])
    heatmap(vecs);
    if mod(i,5) == 0
       w = waitforbuttonpress; 
    end
end
    
    
% %---- perform grad. descent to get optimal params ----
% gammaL = inv(sigmaL + diag(ones(1,nL)*var_init));
% p_sqL = gammaL*sourceTrainL;
% 
% [costs, vari, scales, varis_set, scales_set] = grad_descent(sourceTrainL, numArrays, RTF_train, sigmaL, init_scales, var_init, max_iters, tol, alpha, kern_typ);
% [~,I] = min(sum(costs));
% vari = varis_set(I);
% scales = scales_set(I,:);
% 
% %---- with optimal params estimate test position ----
% sourceTests = [3.3, 3, 1];
% p_hat_ts = zeros(size(sourceTests));
% for i = 1:size(p_hat_ts)
%     sourceTest = sourceTests(i,:);
%     [p_sqL_new, p_hat_ts(i,:)] = test(x, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
%             numMics, sourceTrain, sourceTest, nL, nU, roomSize, vari, T60, c, fs, kern_typ, scales);
% end
% 
% save('mat_outputs/monoTestSource_biMicCircle_5L50U')
% % %  ---- load output data ---- 
% % load('mat_outputs/singleTestSource_biMicCircle_gridL100U')
% 
% %---- plot ----
% plotRoom(roomSize, micsPos, sourceTrain, sourceTests, nL, p_hat_ts);
% title('training grid')