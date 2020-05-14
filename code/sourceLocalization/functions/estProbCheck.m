function [p_align, p_misalign] = estProbCheck(x_tst, transMat, init_var, lambda, gammaL, numMics, numArrays, micsPosNew, scales, RTF_train,rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ)

    %---- Set MRF params ----
    num_iters = 20;
    sub_p_hat_ts = zeros(numArrays, 3); 
    for k = 1:numArrays
        [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPosNew, RTF_train);
        [~,~,sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
            sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
    end
    opt_posteriors = zeros(numArrays,3);
    for it = 1:num_iters
        [~, ~, opt_posteriors_curr] = moveDetectorOpt(x_tst, transMat, init_var, lambda, eMax, 0, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                    rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
        opt_posteriors = opt_posteriors + opt_posteriors_curr;
    end
    opt_posteriors = opt_posteriors./num_iters;
    opt_latents = round(opt_posteriors);
    
    al_probs = opt_posteriors(:,1);
    mis_probs = opt_posteriors(:,2);
    if sum(opt_latents(:,1)) > 0
        p_align = mean(al_probs(opt_latents(:,1) == 1));
    else
        p_align = 0;
    end
    if sum(opt_latents(:,2)) > 0
        p_misalign = mean(mis_probs(opt_latents(:,2) == 1));
    else
        p_misalign = 0;
    end
end