function self_k_new = selfKern(RTF_test,kern_typ,scales,numArrays)
    self_k_new = 0;
    for j = 1:numArrays
        self_k_new = self_k_new + kernel(RTF_test(:,:,j), RTF_test(:,:,j), kern_typ, scales(j));
    end
end