function [upd_sub_p_hat_ts, prob_failure, posteriors] = moveDetectorOpt(x, transMat, init_var, lambda, eMax, thresh, gammaL, numMics, numArrays, micsPos, t, p_fails, sub_p_hat_ts, scales, RTF_train, rirLen, rtfLen,sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ)

    %---- Initialize CRF params ----
    posteriors = zeros(numArrays,3);
    likes = zeros(numArrays,3);
    latents = [ones(numArrays,1), zeros(numArrays,1), zeros(numArrays,1)]+1;

% ----Calculate new estimate based off movement for all subnets and resid. error from stationary time (turns off once estimates settle) ----
    upd_sub_p_hat_ts = zeros(numArrays, 3);
   for k1 = 1:numArrays
        [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, micsPos, RTF_train);
        [~,~,upd_sub_p_hat_ts(k1,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);   
    end
    resids = mean(sub_p_hat_ts-upd_sub_p_hat_ts,2);
    
    %---- Set likelihoods (to be maximized during msg passing calc.) ----
    for k = 1:numArrays
        theta1 = 2*normpdf(resids(k)*2, 0, init_var);
        theta2 = (lambda*exp(-lambda*resids(k)))/(1-exp(-lambda*eMax));
        theta3 = unifrnd(0,eMax);
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
                    mu_alphas(k1,l) = transMat(latents(k2,l),latents(k1,l))*(likes(k2,l)); 
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
        end
        err = norm((posteriors-prev_posts), 2);
        prev_posts = posteriors;
    end

    latents = round(posteriors);
%     MIS = sum(latents(:,2))/(numArrays - sum(latents(:,3))+10e-6);
%     if MIS > .3
%         p_fail = 1;
%     else
%         p_fail = 0;
%     end
%     
    mis_probs = posteriors(:,2);
    if sum(latents(:,2)) > 0
%         prob_failure = mean(mis_probs(latents(:,2) == 1));
        prob_failure = mean(mis_probs(latents(:,2) == 1));
    else
        prob_failure = 0;
    end
end
