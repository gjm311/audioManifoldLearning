
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
%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_biMicCircle_5L50U.mat')

%---- Initialize storage parameters ----
sourceTest = [3.5, 3.5, 1];
iters = 3;
% Q_t = inv(sigmaL+rand*10e-3*eye(size(sigmaL)));
%---- Initialize bayes parameters (via Van Vaerenbergh method) ----
[mu, cov, Q_t] = bayesInit(nL, sourceTrainL, RTF_train, kern_typ, scales, numArrays, vari);
q_t_new = zeros(nL,1);
h_t_new = zeros(nL,1);
thresh = .4;
budgetL = nL;
budgetStr = ceil(nL*1.5);
p_fail = 0;
posteriors = zeros(numArrays, 3);
sub_p_hat_ts = zeros(numArrays, 3); 
self_sub_p_hat_ts = zeros(numArrays, 3);
mvFlag = 0;
rtf_str = zeros(budgetStr,rtfLen, numArrays);
est_str = zeros(budgetStr,3);
numStr = 1; upd = 0;
labels = cell(1,4);
iter = 0;

%---- Initialize moving mic line params ----
movingArray = 1;
init_pauses = 1;
end_pauses = 10;
moving_iters = 4;
numMovePoints = moving_iters + end_pauses + init_pauses;
height = 1;
movingMicsPos = [micLine(micsPos(movingArray,:), [.5,5.5,1], moving_iters), height*ones(moving_iters,1)];
movingMicsPos = [[3 5 1].*ones(init_pauses,3);movingMicsPos];
movingMicsPos = [movingMicsPos; [.5,5.5,1].*ones(end_pauses,3)];


%---- Initialize subnet estimates of training positions ----
randL = randi(numArrays, 1);
holderTest = sourceTrain(randL,:);
for k = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
    [~,~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
        holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
end

mvmnt_truth = [1 0 0; 0 1 0; 0 1 0; 0 1 0];
static_truth = [ones(4,1) zeros(4,1) zeros(4,1)];
ground_truth = zeros(size(posteriors), numMovePoints);
ground_truth(:,:,1:init_pauses) = static_truth;
ground_truth(:,:,init_pauses+1:moving_iters) = mvmnt_truth;
ground_truth(:,:,init_pauses+moving_iters+1:end) = static_truth;

for t = 1:numMovePoints
    %At each iteration, add noise to vary position estimates for more
    %variety when updating. For testing labelled points (to detect
    %movement), we store original precision matrix and only update if we
    %update our subnetwork position estimates; i.e. update when new RTF samples
    %taken.
    iter = iter+1;
    Q_t = Q_t+rand*10e-4*ones(size(Q_t));
    
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
        gammaL = Q_t;
        for k = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPosNew, RTF_train);
            [~,~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
        end
    end
    
    %Check if movement occured. If no movement detected, residual error
    %of position estimate from subnets calculated from initial
    %estimates. If movmement previously detected, we compare to
    %estimate from previous iteration.
    if mvFlag == 0
        [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosNew, iter, p_fail, sub_p_hat_ts, scales, RTF_train,...
        rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
    else
        [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosNew, iter, p_fail, self_sub_p_hat_ts, scales, RTF_train,...
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
        [sub_scales, num_static, RTF_test, p_hat_t, k_t_new] = subEst(Q_t, posteriors, numMics, numArrays, micsPosNew, RTF_train, scales, x, rirLen, rtfLen,...
            sourceTrain, sourceTest, nL, nU, T60, c, fs, kern_typ, roomSize);
        
        %KEY: we store positional estimates taken from sub-network of
        %arrays and test RTF sample measured by ALL arrays (moving
        %included). This way, during update phase, we are updating kernel
        %based off accurate estimates of test position and test RTF sample
        %describing new array network dynamic.
        if latents ~= ones(numArrays, 1)
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
        for s = 1:budgetStr
            if rtf_str(s,:,:) ~= zeros(1,rtfLen,numArrays)
                
                %estimate kernel array between labelled data and test
                k_t_new = zeros(1,nL);
                for i = 1:nL
                    array_kern = 0;
                    for j = 1:numArrays
                        k_t_new(i) = array_kern + kernel(RTF_train(i,:,j), RTF_test(:,:,j), kern_typ, scales(j));
                    end
                end
                [mu,cov,Q_t] = bayesUpd(nL, numArrays, rtf_str(s,:,:), scales, est_str(s,:), kern_typ, k_t_new, Q_t, mu, cov, vari);          
                sourceTrain = [est_str(s,:);sourceTrain];
                RTF_train = cat(1, RTF_test, RTF_train);
                nL = nL+1;
                nD = nD+1;  
            end
        end
        iter = 1;
        mvFlag = 0;
        upd = 1;
        numStr = 1;
        rtf_str = zeros(budgetStr,rtfLen,numArrays);
        est_str = zeros(budgetStr,3);
        [RTF_test, k_t_new, p_hat_t] = test(x, Q_t, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
            numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
            
    %If no movement detected, and estimate position using all arrays and update
    %kernel using params dervied from full network.
    else
        upd = 0;
        [RTF_test, k_t_new, p_hat_t] = test(x, Q_t, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
    end

    %Prune labelled points (if budgetL is maxed out). 
    while nL > budgetL
       [pr_idx, mu, cov, Q_t, sourceTrain] = prune(mu, cov, Q_t, sourceTrain, nL);
       nL = nL - 1;
       RTF_train(pr_idx,:,:) = [];
    end
    
    
%   %Plot
    for p = 1:numArrays
        labels{1,p} = mat2str(round(posteriors(p,:), 2));
    end
    labelPos = [micsPosNew(1:2:end,1), micsPosNew(1:2:end,2)];
    mse = mean((sourceTest-p_hat_t).^2);
    
    f = plotRoom(roomSize, micsPosNew, sourceTrain, sourceTest, nL, p_hat_t);
    title('Detecting Array Movement')
    binTxt = text(3.6,.5, sprintf('Movement Detection Probability: %s \n(t = %g)',num2str(round(p_fail,2)), t-1));
    prbTxt = text(labelPos(:,1),labelPos(:,2), labels, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
    mseTxt = text(.5,.55, sprintf('MSE (Test Position Estimate): %s(m)', num2str(round(mse,2))));
%     %Uncomment below for pauses between iterations only continued by
%     %clicking graph  
    w = waitforbuttonpress;
    if t~=numMovePoints
        delete(binTxt);
        delete(prbTxt);
    end
    clf
    
end   

