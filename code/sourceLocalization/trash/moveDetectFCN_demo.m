clear
%Process Outline:
%Load train data
load('mat_outputs/monoTestSource_biMicCircle_5L50U');
sourceTest = [3.3 3.3 1]; sourceTests = sourceTest;
% sourceTests = sourceTest;
% p_hat_ts = p_hat_t;
addpath ./functions

% % Estimate sub-network test position estimates before movement
sub_p_hat_ts = zeros(numArrays, 3);
for k1 = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, micsPos, RTF_train);
    [~,~,sub_p_hat_ts(k1,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
end

%---- Initialize moving mic line params ----
movingArray = 1;
numMovePoints = 2;
height = 1;
init_pauses = 2;
end_pauses = 2;
movingMicsPos = [micLine(micsPos(movingArray,:), [.5,5.5,1], numMovePoints), height*ones(numMovePoints,1)];
movingMicsPos = [[3 5 1].*ones(init_pauses,3);movingMicsPos];
movingMicsPos = [movingMicsPos; [0.5,5.5,1].*ones(end_pauses,3)];

%---- Initialize CRF params ----
p_fails = zeros(1,numMovePoints);
posteriors = zeros(numArrays,3);
likes = zeros(numArrays,3);
latents = [ones(numArrays,1), zeros(numArrays,1), zeros(numArrays,1)]+1;
transMat = [.75 .2 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .2;
lambda = .2;
eMax = .6;
thresh = .5;


%---- Iterate over all array movements and track probability of movement
%and latent probabilities
for t = 1:numMovePoints+init_pauses+end_pauses
    new_x1 = movingMicsPos(t,1)-.025;
    new_x2 = movingMicsPos(t,1)+.025;
    micsPosNew = [new_x1 movingMicsPos(t,2:3);new_x2 movingMicsPos(t,2:3); micsPos(1+numMics:end,:)];

% ----Calculate new estimate based off movement for all subnets and resid. error from stationary time (turns off once estimates settle) ----
    upd_sub_p_hat_ts = zeros(numArrays, 3);
    for k1 = 1:numArrays
        [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, micsPosNew, RTF_train);
        [~,~,upd_sub_p_hat_ts(k1,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);
    end
    resids = mean(sub_p_hat_ts - upd_sub_p_hat_ts,2);
    sub_p_hat_ts = upd_sub_p_hat_ts;

    %---- Set likelihoods (to be maximized during msg passing calc.) ----
    for k = 1:numArrays
        for l = 1:3
            theta1 = 2*normpdf(resids(k)*100, 0, init_var);
            theta2 = (lambda*exp(-lambda*resids(k)))/(1-exp(-lambda*eMax));
            theta3 = unifrnd(0,eMax);
        end
        likes(k,:) = [theta1 theta2 theta3];
    end

    % ---- For each incoming sample, use CRF to determine prob of latents and detection failure ----
    %Assuming fully connected network, we begin by allowing each latent
    %variable to receive messages from its connected variables to
    %initialize marginal posteriors.
    mu_alphas = zeros(numArrays,3);
    for k1 = 1:numArrays
        for l = 1:3
            for k2 = 1:numArrays
                if k2 ~= k1
                    mu_alphas(k1,l) = transMat(latents(k2,l), latents(k1,l))*(likes(k2,l)); 
                end
            end
            posteriors(k1,l) = likes(k1,l)*mu_alphas(k1,l); 
        end
        posteriors(k1,:) = posteriors(k1,:)./(sum(posteriors(k1,:)));
    end
    
    %Variables send further messages using arbitrary passing strategy.
    mu_alpha_prev = zeros(1,3);
    tol = 10e-3;
    err = inf;
    while err > tol
        if err == inf
            prev_posts = zeros(size(posteriors));
        end
        labels = {};
        for k = 1:numArrays
            for l = 1:3
                if k == 1
                    mu_alpha = transMat(latents(1,l), latents(2,l))*(likes(k,l));
                    mu_alpha_prev(l) = mu_alpha;
                end
                if k == numArrays
                    mu_alpha = transMat(latents(k-1,l), latents(k,l))*mu_alpha_prev(l);
                    mu_alpha_prev(l) = mu_alpha;
                end 
                if and(k>1,k<numArrays)
                    mu_alpha = transMat(latents(k-1,l), latents(k,l))*mu_alpha_prev(l);
                    mu_alpha_prev(l) = mu_alpha;
                end
                posteriors(k,l) = likes(k,l)*mu_alpha;
            end
            posteriors(k,:) = posteriors(k,:)./(sum(posteriors(k,:)));
            labels{end+1} = mat2str(round(posteriors(k,:),2));
        end
        err = norm((posteriors-prev_posts), 2);
        prev_posts = posteriors;
    end
    
    %Calculate probability of misalignment for all subnets (should see higher probabilities of misalignment for subnets with moving array).    
    latents = round(posteriors);
    MIS = sum(latents(:,2))/(numArrays - sum(latents(:,3))+10e-6);
    if MIS > thresh
        p_fails(t) = 1;
    else
        p_fails(t) = 0;
    end
    prob_failure = (1/t)*sum(p_fails);
    latents = latents + 1;
    
    %Plot
    labelPos = [micsPosNew(1:2:end,1), micsPosNew(1:2:end,2)];
    f = plotRoom(roomSize, micsPosNew, sourceTrain, sourceTests, nL, p_hat_ts);
    title('Detecting Array Movement')
    binTxt = text(3.6,.5, sprintf('Movement Detection Probability: %s \n(t = %g)',num2str(prob_failure), t-1));
    prbTxt = text(labelPos(:,1),labelPos(:,2), labels, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

    %Uncomment below for pauses between iterations only continued by
    %clicking graph  
    w = waitforbuttonpress;
    if t~=numMovePoints+init_pauses+end_pauses
        delete(binTxt);
        delete(prbTxt);
    end
end
    



