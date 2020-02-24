clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_biMicCircle_5L50U.mat')
gammaL = inv(sigmaL+rand*10e-3*eye(size(sigmaL)));

%TRAINING
%Estimate labelled positions for all individual arrays and estimate error.
distributions = zeros(numArrays, nL);

lab_p_hats = zeros(numArrays, nL, 3);
lab_errs = zeros(numArrays, nL);

for k = 1:numArrays
    for nl = 1:nL
        holderTest = sourceTrain(nl,:);
        curr_mics = micsPos(k,:);
        scale = scales(k);
        trRTF = RTF_train(:,:,k);
        [~,~, lab_p_hats(k,nl,:)] = test(x, gammaL, trRTF, curr_mics, rirLen, rtfLen, 1, numMics, sourceTrain,...
            holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, scale); 
        lab_errs(k,nl) = norm(lab_p_hats(k,nl,:)-sourceTrain(nl,:));
    end
end

%Estimate unlabelled positions with each individual array. Via Dirichlet
%process, define distribution for each array whereby we add an unlabelled
%estimate to a given 'error table' based on similarity to positional
%estimate and popularity of table.
dir_pro = @(n_alph, s_j, n) n_alph/(s_j+n-1);
n_alph = ones(nL,1);

for k = 1:numArrays
    for nu = nL+1:nD
        holderTest = sourceTrain(nu,:);
        curr_mics = micsPos(k,:);
        scale = scales(k);
        trRTF = RTF_train(:,:,k);
        [~,~, unl_p_hat] = test(x, gammaL, trRTF, curr_mics, rirLen, rtfLen, 1, numMics, sourceTrain,...
            holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, scale);
        est_comp = squareform(pdist([unl_p_hat;lab_p_hats]));
        unl_errs = est_comp(1,2:end);
        unl_probs = dir_pro(n_alph, unl_errs, nL+nu);
        [~,max_idx] = max(unl_probs);
        distributions(max_idx) = distributions(max_idx)+1;
    end
end

%4) Interpolate weights to fit labelled points to true position.


%TEST
%Estimate test RTF position and compare residual to probability (based off 


%Estimate unlabelled positions for all individual microphones. Based off
%estimate, increase probability of that particular residual error 