function [cov,covL,K] = trCovEstK(nL, nD, numArrays, rtfs, kernTyp, scales)
    K = zeros(numArrays, nD, nD);
    
    for a = 1:numArrays
        for d1 = 1:nD
            for d2 = 1:nD
                K(a,d1,d2) = kernel(rtfs(d1,:,a), rtfs(d2,:,a), kernTyp, scales(a));
            end
        end
    end
    
    cov = zeros(nD,nD);
    for a1 = 1:numArrays
        for a2 = 1:numArrays
            cov = cov + reshape(K(a1,:,:).*K(a2,:,:),[nD,nD]);
        end
    end
    cov = (1/numArrays^2).*cov;
    covL = cov(1:nL,1:nL);
end