%{
This program is able to detect movement of an array based off of a
misalignment recognition algorithm. The algorithm utilizes MRFs with Fully
Connected latent variables to predict misalignment from sensor
measurements. Here the input is the residual error between an estimated
position by subsets of the overall network and previous estimates. As one would expect, 
if the probability of misalignment is high for all subnets but one, this is 
likely the node that is being moved. More details can be found in
"Misalignment Recognition Using MRFs with Fully connected Latent Variables
for Detecting Localization Failures" by Naoki Akai, Luis Yoichi Morales et
al.
%}

clear
addpath ./functions
%Load train data
load('mat_outputs/monoTestSource_biMicCircle_5L300U');

% % Estimate sub-network test position before node movement
sub_p_hat_ts = zeros(numArrays, 3);
for k = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
    [~,sub_p_hat_ts(k,:)] = test(x, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, vari, T60, c, fs, kern_typ, subscales);  
end

%---- Initialize moving mic line params ----
movingArray = 1;
numMovePoints = 5;
height = 1;
% pauses = 1;
movingMicsPos = [micLine(micsPos(movingArray,:), [.5,5.5,1], numMovePoints), height*ones(numMovePoints,1)];
% movingMicsPos = [[3 5 1].*ones(pauses,3);movingMicsPos];
p_fails = zeros(1,numMovePoints);

%---- Initialize CRF params ----
posteriors = zeros(numArrays,3);
latents = [ones(numArrays,1), zeros(numArrays,1), zeros(numArrays,1)]+1;
transMat = [.7 0.25 .05; .15 .8 .05; .5 .45 .05];
var = .5;
lambda = .3;
eMax = .6;
thresh = .5;

%---- Iterate over all array movements and track probability of movement
%and latent probabilities
for t = 1:numMovePoints
    new_x1 = movingMicsPos(t,1)-.1;
    new_x2 = movingMicsPos(t,1)+.1;
    micsPosNew = [new_x1 movingMicsPos(t,2:3);new_x2 movingMicsPos(t,2:3); micsPos(1+numMics:end,:)];

% ----Calculate new position estimates for each subnetwork based off movement for all subnets and resid. error based off estimate from previous iteration ----
    upd_sub_p_hat_ts = zeros(numArrays, 3);
    for k = 1:numArrays
        [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPosNew, RTF_train);
        [~,upd_sub_p_hat_ts(k,:)] = test(x, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, vari, T60, c, fs, kern_typ, subscales);   
    end
    resids = mean(sub_p_hat_ts - upd_sub_p_hat_ts,2);
    sub_p_hat_ts = upd_sub_p_hat_ts;

    %iterate for incoming samples  for each subnet and use CRF to determine prob. of
    %detection failure
    mu_alpha_prev = zeros(1,3);
    mu_beta_prev = zeros(1,3);
    labels = {};
    for k = 1:numArrays
        for l = 1:3
            if l == 1
                theta = 2*normpdf(resids(k), 0, var);
            end
            if l == 2
                theta = (lambda*exp(-lambda*resids(k)))/(1-exp(-lambda*eMax));
            end
            if l == 3
                theta = unifrnd(0,eMax);
            end

            if k == 1
                mu_alpha = transMat(latents(1,l), latents(2,l))*(theta);
                mu_alpha_prev(l) = mu_alpha;
                mu_beta =  transMat(latents(numArrays,1), latents(numArrays-1,1));
                mu_beta_prev(l) = mu_beta;
            end
            if k == numArrays
                mu_alpha = transMat(latents(k-1,l), latents(k,l))*mu_alpha_prev(l);
                mu_alpha_prev(l) = mu_alpha;
                mu_beta = transMat(latents(1,l), latents(2,l))*mu_beta_prev(l);
                mu_beta_prev(l) = mu_beta;
            end 
            if and(k>1,k<numArrays)
                mu_alpha = transMat(latents(k-1,l), latents(k,l))*mu_alpha_prev(l);
                mu_alpha_prev(l) = mu_alpha;
                mu_beta = transMat(latents(numArrays-k+1,l), latents(numArrays-k,l))*mu_beta_prev(l);
                mu_beta_prev(l) = mu_beta;
            end
            posteriors(k,l) = theta*mu_alpha*mu_beta;
        end
        posteriors(k,:) = posteriors(k,:)./(sum(posteriors(k,:)));
        labels{end+1} = mat2str(round(posteriors(k,:),2));
    end
    %Calculate probability of misalignment (should see higher probs. of
    %misalignment (2nd value in vector) for subnets containg moving array).
    
    latents = round(posteriors);
    MIS = sum(latents(:,2))/(numArrays - sum(latents(:,3))+10e-6);
    if MIS > thresh
        p_fails(t) = 1;
    else
        p_fails(t) = 0;
    end
    prob_failure = (1/t)*sum(p_fails);
    latents = latents + 1;
    
    labelPos = [micsPosNew(1:2:end,1), micsPosNew(1:2:end,2)];
    f = plotRoom(roomSize, micsPosNew, sourceTrain, sourceTests, nL, p_hat_ts);
    title('Detecting Array Movement')
    binTxt = text(3.6,.5, sprintf('Movement Detection Probability: \n%s', num2str(prob_failure)));
    prbTxt = text(labelPos(:,1),labelPos(:,2), labels, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
    
    
    %Uncomment below for pauses between iterations only continued by
    %clicking graph  
    w = waitforbuttonpress;
    if t~=numMovePoints
        delete(binTxt);
        delete(prbTxt);
    end
end
    



