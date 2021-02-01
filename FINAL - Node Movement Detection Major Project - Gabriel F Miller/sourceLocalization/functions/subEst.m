function [sub_scales, num_static, RTF_test, p_hat_t, k_t_new] = subEst(gammaL, posteriors, numMics, numArrays, micsPos, RTF_train, scales, x, rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, T60, c, fs, kern_typ, roomSize)
    %This function finds probabilisitcally static arrays and if this is all
    %arrays (i.e. all arrays static), we compute new RTF sample (based off positioning of all arrays),
    %new test kernel vector (k_t_new) and test position estimate. IF one or
    %more arrays are moving, we update test kernel vector and estimate test
    %position based off sub-network of static arrays. RTF_test of course is
    %still calculated based off positioning of all arrays.
    
    RTF_test = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTest, roomSize, T60, rirLen, c, fs);

    misaligned_idxs = find(round(posteriors(:,1)) == 1);
    misalignedMic_idxs = misaligned_idxs;
    for ms = 1:numMics-1
        misalignedMic_idxs = sort(vertcat(misalignedMic_idxs*numMics,misalignedMic_idxs*numMics-ms));
    end
    
    num_static = numArrays - size(misaligned_idxs,1);
    if num_static == numArrays
       sub_RTF_train = RTF_train; 
       sub_scales = scales; 
       num_static = 0;
       [RTF_test,k_t_new,p_hat_t] = test(x, gammaL, RTF_train, micsPos, rirLen, rtfLen, size(sub_RTF_train,3),...
                numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, sub_scales); 
    else
        drop_idxs = zeros(1,numArrays);
        dropMic_idxs = zeros(1,numArrays*numMics);
        drop_idxs(misaligned_idxs) = 1;
        dropMic_idxs(misalignedMic_idxs) = 1;
        sub_RTF_train = RTF_train(:,:,~drop_idxs);
        sub_micsPos = micsPos(~dropMic_idxs,:);
        sub_scales = scales(~drop_idxs);
        [~,k_t_new, p_hat_t] = test(x, gammaL, sub_RTF_train, sub_micsPos, rirLen, rtfLen, (numArrays-sum(drop_idxs)),...
                numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, sub_scales);  
    end
end