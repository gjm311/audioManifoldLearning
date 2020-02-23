function [RTF_test, p_hat_t] = test(x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales)
    %---- update estimate based off test ----
    RTF_test =  rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTest, roomSize, T60, rirLen, c, fs);
   
    %estimate kernel array between labelled data and test
    k_t_new = zeros(1,nL);
    for i = 1:nL
        for j = 1:numArrays
            k_t_new(i) = k_t_new(i) + kernel(RTF_train(i,:,j), RTF_test(:,:,j), kern_typ, scales(j));
        end
    end

    %Uncomment for update method via Bracha's paper (vs. update done in
    %bayesUpd.m from Vaerenbergh's paper.
    nD = nL+nU;
    sourceTrainL = sourceTrain(1:nL,:);
    gammaL_new = gammaL - ((gammaL*(k_t_new'*k_t_new)*gammaL)./(numArrays^2+k_t_new*gammaL*k_t_new'));
    gammaL_new = gammaL;
    p_sqL_new = gammaL_new*sourceTrainL;

    %estimate test covariance estimate
    sigma_Lt = tstCovEst(nL, nD, numArrays, RTF_train, RTF_test, kern_typ, scales); 
%     sigmaLt_new = sigma_Lt + (1/numArrays).*k_t_new;
    p_hat_t = sigma_Lt*p_sqL_new;

end