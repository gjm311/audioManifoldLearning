clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech
%{
This program looks to build off the estimation process used in
sourceLocal_main.m (i.e. based off a semi-supervised manifold learning
approach), but rather than assume uniform contribution by each array, we
weigh the contribution based on a so-called error similarity profile. 
In particular, we measure and store the error in predicting
labelled training positions (assuming uniform contribution). We then
estimate each test position, and compare the prediction to that of the
labelled positions. Based off the similarity (i.e. euclidean distance) 
between predictions, we build a distribution based off a Dirichlet process (DP). 
That is to say based off the prediction similarity between the labelled positions, 
and unlabelled (which were sampled uniformly at random) we increase the 
frequency of a given error grouping which coincides with the number of labelled predictions. 
Because the distribution is measured via a DP, we are also able to factor in 
the popularity of a given error class to account for potential disparity in each array's ability to localize due to
positioning in a given room environment as well as the room's given parameters 
(i.e. reverberation, noise level (like an array positioned close to a fan), etc.).


%}

% %---- load training data (check mat_trainParams for options)----
% load('mat_outputs/monoTestSource_biMicCircle_5L50U_monoNode.mat')
% gammaL = inv(sigmaL+rand*10e-3*eye(size(sigmaL)));
% 
% %TRAINING
% %Estimate labelled positions for all individual arrays and estimate error.
% distributions = ones(numArrays, nL);
% 
% lab_p_hats = zeros(nL, 3, numArrays);
% lab_errs = zeros(numArrays, nL);
% 
% for k = 1:numArrays
%     for nl = 1:nL
%         holderTest = sourceTrain(nl,:);
%         curr_mics = micsPos((k-1)*numMics+1:k*numMics,:);
%         curr_scales = ones(1,numMics).*scales(k);
%         trRTF = reshape(RTF_train_monoArrays(k,:,:,:), [nD, rtfLen, numMics]);
%         [~,~, lab_p_hats(nl,:,k)] = test(x, gammaL, trRTF, curr_mics, rirLen, rtfLen, 1, numMics, sourceTrain,...
%             holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, curr_scales); 
%         lab_errs(k,nl) = norm(lab_p_hats(nl,:,k)-sourceTrain(nl,:));
%     end
% end
% 
% %Estimate unlabelled positions with each individual array. Via Dirichlet
% %process, define distribution for each array whereby we add an unlabelled
% %estimate to a given 'error table' based on similarity to positional
% %estimate and popularity of table.
% dir_pro = @(n_alph, s_j, n, scales) (scales.*n_alph)./(s_j+n-1);
% 
% for k = 1:numArrays
%     for nu = nL+1:nD
%         holderTest = sourceTrain(nu,:);
%         curr_mics = micsPos((k-1)*numMics+1:k*numMics,:);
%         curr_scales = ones(1,numMics).*scales(k);
%         trRTF = reshape(RTF_train_monoArrays(k,:,:,:), [nD, rtfLen, numMics]);
%         [~,~, unl_p_hat] = test(x, gammaL, trRTF, curr_mics, rirLen, rtfLen, 1, numMics, sourceTrain,...
%             holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, curr_scales);
%         est_comp = squareform(pdist([unl_p_hat;lab_p_hats(:,:,k)]));
%         unl_errs = est_comp(1,2:end);
%         unl_probs = dir_pro(lab_errs(k,:), unl_errs, nD, opt_dp_alphs(k,nl));
%         distributions(k,unl_errs == max(unl_errs)) = distributions(k,unl_errs == max(unl_errs))+1;
%     end
% end
% distributions = distributions./nD;
% 
% %---- Build kernel covariance matrices for all labelled points ---- 
% sigmaLs = zeros(nL,nL,nL);
% gammaLs = zeros(nL,nL,nL);
% scale_set = zeros(nL,numArrays)
% for nl = 1:nL
%     [~,sigmaLs(nl,:,:)] = wTrCovEst(distributions(:,nl), nL, nD, numArrays, RTF_train, kern_typ, scales);
%     gammaLs(nl,:,:) =  inv(reshape(sigmaLs(nl,:,:),[nL,nL])+rand*10e-3*eye(size(sigmaL)));
%%     scale_set(nl,:) =  
% end

for nl = 1:nL
   scale_set(nl,:) = grad_descent()
end

load('mat_trainParams/biMicCircle_5L50U_monoNode.mat')
RTF_train_monoArrays = RTF_train;
load('mat_outputs/monoTestSource_biMicCircle_5L50U.mat')
load('mat_outputs/monoTestSource_biMicCircle_5L50U_monoNode.mat')
save('mat_outputs/monoTestSource_biMicCircle_5L50U_monoNode.mat')

%TEST
% ---- est. tst position using kern. cov. w/uniform array contribution ----
[~,~, p_hat_t] = test(x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
tRes = squareform(pdist([p_hat_t;mean(lab_p_hats,3)]));
[~, min_tRes_idx] = min(tRes(1,2:end));
[~,~, p_hat_t] = test(x, reshape(gammaLs(min_tRes_idx,:,:),[nL,nL]), RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);

