function [aucs, out_potential] = estProbCheck(radius_curr, lambdas, init_vars, radii,sourceTrain,mic_ref, transMat,wavs,T60,modelMean,modelSd, RTF_train,scales,gammaL,nL, nU,rirLen, rtfLen,c, kern_typ,roomSize,radiusU, ref, numArrays, micsPos, numMics, fs)
    out_potential = zeros(3,3);
    out_potential(3,:) = ones(1,3).*(1/3);
   
    %---- Set MRF params ----
    eMax = .3;
    threshes = 0:.25:1;
    num_iters = 5;
    numVaris = size(init_vars,2);
    numLams = size(lambdas,2);

    aucs = zeros(numLams, numVaris); 
    posteriors = zeros(numLams,numVaris,numArrays,3);
    
    for lam = 1:numLams
        lambda = lambdas(lam);

        for v = 1:numVaris
            init_var = init_vars(v);           

            [tpr_out, fpr_out, posteriors(lam,v,:,:)] = potentialOpt(radius_curr,sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
            aucs(lam,v) = trapz(flip(fpr_out),flip(tpr_out));
        end
    end
    
    [opt_lam_pos,opt_var_pos] = find(aucs == max(max(aucs)));
    opt_posteriors = posteriors(opt_lam_pos,opt_var_pos,:,:);
    opt_latents = round(opt_posteriors);
    
    al_probs = opt_posteriors(:,1);
    mis_probs = opt_posteriors(:,2);
    if sum(opt_latents(:,1)) > 0
        out_potential(1,1) = mean(al_probs(opt_latents(:,1) == 1));
    else
        out_potential(1,1) = 0;
    end
    if sum(opt_latents(:,2)) > 0
        out_potential(2,2) = mean(mis_probs(opt_latents(:,2) == 1));
    else
        out_potential(2,2) = 0;
    end
    out_potential(1,3) = 1-out_potential(1,1);
    out_potential(2,3) = 1-out_potential(2,2);
end