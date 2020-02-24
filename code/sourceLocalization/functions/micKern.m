function k_t_new = micKern(RTF_train, RTF_test, kern_typ, scales, nD, numArrays)
    k_t_new = zeros(1,nD);
    for i = 1:nD
        for j = 1:numArrays
            k_t_new(i) = k_t_new(i) + kernel(RTF_train(i,:,j), RTF_test(:,:,j), kern_typ, mean(scales));
        end
    end
end
