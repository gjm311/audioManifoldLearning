%{
This program iterates over time and at each iteration, determines if movement in the network 
occurs by comparing stored sub-network position estimates of all labelled training position 
with online estimates. Movement detection is done via a fully connected MRF utilizing residual 
errors based off sub-network positional estimates with those stored during training. If no 
movement occurs, no changes to the inverse covariance matrix (key in estimating source positions) 
occur, but position estimates are stored. If movement occurs, we estimate the position of the 
source using the sub-network that is still in position and movement detection is done 
with residual errors calculated based off positional estimates from previous iterations. This 
is done to ensure that once the moving array stops, we are not comparing estimates with those 
stored for the original array network setup. Once movement is no longer detected, we use the 
stored test position estimates as new labelled positions, and update the covariance matrix 
in a Bayesian manner to adjust for the new array positioning. We also incorporate a labelled 
budget, which if met, we prune our parameters. I.e. for the sake of saving storage space, 
and also incorporating a 'forgetting mechanism', such that only relevant RTF samples are used, 
we remove samples that yield the smallest information loss when removed.
%}

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

%---- load training data (check trainRoom_sim.m for details)----
load('mat_outputs/monoTestSource_biMicCircle_5L50U.mat')

%---- Initialize general parameters ----
sourceTest = [3.3, 3.3, 1];
QL = gammaL;
% [mu, cov, Q_D] = bayesInit(nD, sourceTrain, RTF_train, kern_typ, scales, numArrays, vari);
% Q_t = Q_D(1:nL,1:nL);

%---- Initialize bayes parameters (via Van Vaerenbergh method) ----
q_t_new = zeros(nL,1);
h_t_new = zeros(nL,1);
thresh = .4;
budgetL = nL;
budgetD = nD;
budgetStr = 10;
p_fail = 0;
posteriors = zeros(numArrays, 3);
sub_p_hat_ts = zeros(numArrays, 3); 
self_sub_p_hat_ts = zeros(numArrays, 3);
sub_sigma_Lts = zeros(numArrays, nL);
mvFlag = 0;
rtf_str = zeros(budgetStr,rtfLen, numArrays);
est_str = zeros(budgetStr,3);
numStr = 1; upd = 0;
labels = cell(1,4);
p_fail_count = 0;

%---- Initialize moving mic array params ----
movingArray = 1;
init_pauses = 1;
end_pauses = 10;
moving_iters = 4;
numMovePoints = moving_iters + end_pauses + init_pauses;
height = 1;
startPos = micsPos(movingArray,:);
endPos = [.5,5.5,1];
movingMicsPos = [micLine(startPos, endPos, moving_iters), height*ones(moving_iters,1)];
movingMicsPos = [[3 5 1].*ones(init_pauses,3);movingMicsPos];
movingMicsPos = [movingMicsPos; [.5,5.5,1].*ones(end_pauses,3)];

%---- Initialize subnet estimates of training positions ----
randL = randi(numArrays, 1);
holderTest = sourceTrain(randL,:);
for k = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
    [~, sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
        holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
end


for t = 1:numMovePoints
    %At each iteration, add noise to vary position estimates for more
    %variety when updating. For testing labelled points (to detect
    %movement), we store original precision matrix and only update if we
    %update our subnetwork position estimates; i.e. update when new RTF samples
    %taken.
    p_fail_count = p_fail_count+1;
%     QL = QL+rand*10e-4*ones(size(QL));
    
    new_x1 = movingMicsPos(t,1)-.025;
    new_x2 = movingMicsPos(t,1)+.025;
    micsPosNew = [new_x1 movingMicsPos(t,2:3);new_x2 movingMicsPos(t,2:3); micsPos(1+numMics:end,:)];
    
    %Estimate position for all labelled points for all subnetworks and take
    %average probability of movement for each labelled point.
    %Estimate training position for each subnetwork. Calculated when
    %movement stops and new labelled RTF samples added. 
    if upd == 1
        randL = randi(numArrays,1);
        holderTest = sourceTrain(randL,:);
        gammaL = QL;
        for k = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPosNew, RTF_train);
            [~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
        end
    end
    
    %Check if movement occured. If no movement detected, residual error
    %of position estimate from subnets calculated from initial
    %estimates. If movmement previously detected, we compare to
    %estimate from previous iteration.
    if mvFlag == 0
        [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosNew, p_fail_count, p_fail, sub_p_hat_ts, scales, RTF_train,...
        rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
    else
        [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosNew, p_fail_count, p_fail, self_sub_p_hat_ts, scales, RTF_train,...
        rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
    end
    latents = round(posteriors);
    
    %IF movement detected, estimate test position using subnet (w/o moving array).
    %Update parameters and covariance via Bayes with params derived from subnetwork.
    if p_fail > thresh
        if mvFlag == 0
           mvFlag = 1;
        end
        upd = 0;
        %Note, in practice we should continue using subnet until p_fail
        %dips below thresh, but here (for illustrative purposes) we switch
        %once posterior probabilities favors alignment again which should
        %yield a worse estimate as we are utilizing RTFs of original
        %dynamic though test RTF sample reflects new array position.
        [sub_scales, num_static, RTF_test, p_hat_t] = subEst(QL, posteriors, numMics, numArrays, micsPosNew, RTF_train, scales, x, rirLen, rtfLen,...
            sourceTrain, sourceTest, nL, nU, T60, c, fs, kern_typ, roomSize);
        
        p_hat_t = p_hat_t + rand(size(p_hat_t))*10e-3;
        %KEY: we store positional estimates taken from sub-network of
        %arrays and test RTF sample measured by ALL arrays (moving
        %included). This way, during update phase, we are updating kernel
        %based off accurate estimates of test position and test RTF sample
        %describing new array network dynamic.
        if sum(latents(:,1)) ~= numArrays
            if numStr > budgetStr
                rtf_str(1,:,:) = RTF_test;
                rtf_str = circshift(rtf_str, -1, 1);
                est_str(1,:) = p_hat_t;
                est_str = circshift(est_str, -1, 1);
            else
                rtf_str(numStr,:,:) = RTF_test;
                est_str(numStr,:) = p_hat_t;
                numStr = numStr + 1;
            end
        end
               
        
    %Right after movement is no longer detected (i.e. right after probability of
    %movement dips below threshold), update covariance matrix using saved
    %up test position estimates.
    elseif mvFlag == 1
        nonZero_idxs = find(~all(est_str==0,2));      
        sourceTrain = cat(1,sourceTrain, est_str(nonZero_idxs,:));
        RTF_train = cat(1, RTF_train, rtf_str(nonZero_idxs,:,:));
        nD = nD+size(est_str(nonZero_idxs,:),1);  
        [mu, cov, QD] = bayesInit(nD, sourceTrain, RTF_train, kern_typ, scales, numArrays, vari);
%                 [mu,cov,Q_t] = bayesUpd(nL, numArrays, rtf_str(s,:,:), scales, est_str(s,:), kern_typ, k_t_str(s,:), Q_t, mu, cov, vari);             
        
        
        %Prune all RTF points (if budget is maxed out). 
        while nD > budgetD
           [pr_idx, mu, cov, QD, sourceTrain] = prune(mu, cov, QD, sourceTrain, nD);
           nD = nD - 1;
           RTF_train(pr_idx,:,:) = [];
        end
        QL = QD(1:nL,1:nL);        
        p_fail_count = 1;
        mvFlag = 0;
        upd = 1;
        numStr = 1;
        rtf_str = zeros(budgetStr,rtfLen,numArrays);
        est_str = zeros(budgetStr,3);
        [RTF_test, p_hat_t] = test(x, QL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
            numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
            
    %If no movement detected, and estimate position using all arrays and update
    %kernel using params dervied from full network.
    else
        upd = 0;
        [RTF_test, p_hat_t] = test(x, QL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
    end    
    
%   %Plot
    for p = 1:numArrays
        labels{1,p} = mat2str(round(posteriors(p,:), 2));
    end
    labelPos = [micsPosNew(1:2:end,1), micsPosNew(1:2:end,2)];
    eudist = norm(sourceTest-p_hat_t);
    
    f = plotRoom(roomSize, micsPosNew, sourceTrain, sourceTest, nL, p_hat_t);
    title('Detecting Array Movement')
    binTxt = text(3.6,.5, sprintf('Movement Detection Probability: %s \n(t = %g)',num2str(round(p_fail,2)), t-1));
    prbTxt = text(labelPos(:,1),labelPos(:,2), labels, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
    mseTxt = text(.5,.55, sprintf('MSE (Test Position Estimate): %s(m)', num2str(round(eudist,2))));
%     %Uncomment below for pauses between iterations only continued by
%     %clicking graph  
    w = waitforbuttonpress;
    if t~=numMovePoints
        delete(binTxt);
        delete(prbTxt);
    end
    clf
    
end   
