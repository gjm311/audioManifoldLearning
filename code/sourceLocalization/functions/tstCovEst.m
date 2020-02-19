function cov = tstCovEst(nL, nD, numArrays, trRtfs, tstRtf, kernTyp, scales)
    cov = zeros(1,nL); 
    
    for r = 1:nL
        kern = 0;
        for i = 1:nD
            for q = 1:numArrays
                for w = 1:numArrays
                    kernQ = kernel(tstRtf(:,:,q), trRtfs(i,:,q), kernTyp, scales(q));
                    kernW = kernel(tstRtf(:,:,w), trRtfs(i,:,w), kernTyp, scales(w));
                    kern = kern + kernQ*kernW;
                end
            end
        end
        cov(1,r) = (1/numArrays^2)*kern;     
    end 
end