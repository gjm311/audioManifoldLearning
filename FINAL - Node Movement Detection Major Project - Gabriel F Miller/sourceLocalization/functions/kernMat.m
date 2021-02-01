function K = kernMat(rtfs, nL, nD, numArrays, kernTyp, scales)

    K = zeros(numArrays, nL, nL);

    for q = 1:numArrays
        for k = 1:nD
            for i = 1:nL
                for j = 1:nL
                    K_i = kernel(rtfs(i,:,q), rtfs(k,:,q), kernTyp, scales(q));
                    K_j = kernel(rtfs(j,:,q), rtfs(k,:,q), kernTyp, scales(q)); 
                    K(q,i,j) = K_i*K_j;
                end
            end
        end
    end
    K = reshape(mean(K,1),[nL,nL]);
end