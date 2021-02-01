function [vari, scales] = x_val(nL, nD, numArrays, RTF_train, kern_typ, sourceTrainL)
    init_scales = 1*ones(1,numArrays);
    var_init = 0;
    tol = 10e-3;
    alpha = 10e-3;
    max_iters = 100;

    [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, init_scales);

    %---- perform grad. descent to get optimal params ----
    gammaL = inv(sigmaL + diag(ones(1,nL)*var_init));
    p_sqL = gammaL*sourceTrainL;

    [costs,~,~, varis_set, scales_set] = grad_descent(sourceTrainL, numArrays, RTF_train, sigmaL, init_scales, var_init, max_iters, tol, alpha, kern_typ);
    [~,I] = min(sum(costs));
    vari = varis_set(I);
    scales = scales_set(I,:);
end
