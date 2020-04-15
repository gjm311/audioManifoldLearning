function [tprs, fprs] = paramOpt(sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs)

    num_radii = size(radii, 2);
    num_threshes = size(threshes, 2);
   
    tprs = zeros(1,num_threshes);
    fprs = zeros(1,num_threshes);
    
    for thr = 1:num_threshes
        thresh = threshes(thr);
        [tps,fps,tns,fns] = paramThreshOpt(num_radii, thresh, sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
        
        tprs(thr) = tps/(tps+fns);
        fprs(thr) = fps/(fps+tns);        
    end
    
end