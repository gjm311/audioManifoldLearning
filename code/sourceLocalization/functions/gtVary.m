function [mrf_res] = gtVary(thresh, sourceTrain, wavs, gammaL, T60, gt, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs)
    num_radii = size(radii, 2);
    
    tp_ch_curr = 0;
    fp_ch_curr = 0;
    tn_ch_curr = 0;
    fn_ch_curr = 0;

    for riter = 1:num_radii
        radius_mic = radii(riter);

        for iters = 1:num_iters      
            %randomize tst source, new position of random microphone (new
            %position is on circle based off radius_mic), random rotation to
            %microphones, and random sound file (max 4 seconds).
            sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
            movingArray = randi(numArrays);
            [~, micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);

            rand_wav = randi(25);
            try
                file = wavs(rand_wav+2).name;
                [x_tst,fs_in] = audioread(file);
                [numer, denom] = rat(fs/fs_in);
                x_tst = resample(x_tst,numer,denom);
                x_tst = x_tst';  
            catch
                continue
            end    

          %---- Initialize subnet estimates of training positions ----
            sub_p_hat_ts = zeros(numArrays, 3); 
            for k = 1:numArrays
                [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
                [~,~,sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
            end
            [~, p_fail, ~] = moveDetectorOpt(x_tst, transMat, init_var, lambda, eMax, thresh, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                    rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            
            if radius_mic >= gt             
                if p_fail >= thresh
                    tp_ch_curr = tp_ch_curr + 1;
                else
                    fn_ch_curr = fn_ch_curr + 1;
                end
            end
            if radius_mic < gt   
                if p_fail >= thresh
                    fp_ch_curr = fp_ch_curr + 1;
                else
                    tn_ch_curr = tn_ch_curr + 1;
                end
   
            end
        end
    end
    mrf_res = [tp_ch_curr fp_ch_curr tn_ch_curr fn_ch_curr];
%     nai_res = [mean_tp_ch_curr mean_fp_ch_curr mean_tn_ch_curr mean_fn_ch_curr]./num_iters*num_radii;
%     sub_res = [sub_tp_ch_curr sub_fp_ch_curr sub_tn_ch_curr sub_fn_ch_curr]./num_iters*num_radii;
    
end