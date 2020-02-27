clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
Here we optimize scalar hyper-parameter w/respect to the Dirichlet process (DP).
Process is used in nearestNeighbor.m where we look to weigh the
contribution each array gives in calculating the kernel covariance matrix
based on the residual errors output in estimating labelled positions. This
leads to an error similarity distribution we build via the DP and 
unlabelled training point estimates.
%}

%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_ biMicCircle_5L50U_monoNode.mat')
gammaL = inv(sigmaL+rand*10e-3*eye(size(sigmaL)));

%TRAINING
%Estimate labelled positions for all individual arrays and estimate error.
num_iters = 100;
dir_alph = zeros(num_iters,numArrays,nL);
sim_count = zeros(1,num_iters);

for iter = 1:num_iters
    distributions = ones(numArrays, nL);
    lab_p_hats = zeros(nL, 3, numArrays);
    lab_errs = zeros(numArrays, nL);

    for k = 1:numArrays
        dir_alph(iter,k,:) = rand(1,nL );
        for nl = 1:nL
            holderTest = sourceTrain(nl,:);
            curr_mics = micsPos((k-1)*numMics+1:k*numMics,:);
            curr_scales = ones(1,numMics).*scales(k);
            trRTF = reshape(RTF_train(k,:,:,:), [nD, rtfLen, numMics]);
            [~,~, lab_p_hats(nl,:,k)] = test(x, gammaL, trRTF, curr_mics, rirLen, rtfLen, 1, numMics, sourceTrain,...
                holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, curr_scales); 
            lab_errs(k,nl) = norm(lab_p_hats(nl,:,k)-sourceTrain(nl,:));
        end
    end

    %Estimate unlabelled positions with each individual array. Fine tune DP
    %scalars based on ability to predict proximity to a given array based
    %on probability of belonging to a given error class; i.e. based on
    %whether or not similarity in error probabilistically implicates closest mic.
    dir_pro = @(n_alph, s_j, n, scales) (scales.*n_alph)./(s_j+n-1);

    for k = 1:numArrays
        for nu = nL+1:nD
            %estimate ground truth closest mic to unlabelled point
            micDist = squareform(pdist([sourceTrain(nu,:); micsPos]));
            micDist = micDist(1,2:end);
            [~,groundTruth_idx] = min(micDist);
            
            holderTest = sourceTrain(nu,:);
            curr_mics = micsPos((k-1)*numMics+1:k*numMics,:);
            curr_scales = ones(1,numMics).*scales(k);
            trRTF = reshape(RTF_train(k,:,:,:), [nD, rtfLen, numMics]);
            [~,~, unl_p_hat] = test(x, gammaL, trRTF, curr_mics, rirLen, rtfLen, 1, numMics, sourceTrain,...
                holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, curr_scales);
            est_comp = squareform(pdist([unl_p_hat;lab_p_hats(:,:,k)]));
            unl_errs = est_comp(1,2:end);
            unl_probs = dir_pro(lab_errs(k,:), unl_errs, nD, dir_alph(iter,k,:));
            [~,maxProb_idx] = max(unl_probs);
            if groundTruth_idx == maxProb_idx
                sim_count(iter) = sim_count(iter)+1;
            end
        end
    end
end

% [~,opt_idx] = max(sim_count);
% opt_dp_alphs = reshape(dir_alph(opt_idx,:,:), [numArrays,nL]);
% save('mat_outputs/monoTestSource_biMicCircle_5L50U_monoNode.mat');