function [opt_tps, opt_fps, opt_tns, opt_fns, max_auc] = paramOpt(sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs)

    num_radii = size(radii, 2);
    num_threshes = size(threshes, 2);
   
    tps = zeros(1,num_threshes);
    fps = zeros(1,num_threshes);
    tns = zeros(1,num_threshes);
    fns = zeros(1,num_threshes);
    tprs = zeros(1,num_threshes);
    fprs = zeros(1,num_threshes);
    
    for thr = 1:num_threshes
        thresh = threshes(thr);

        for riter = 1:num_radii
            radius_mic = radii(riter);
            
            tp_ch_curr = 0;
            fp_ch_curr = 0;
            tn_ch_curr = 0;
            fn_ch_curr = 0;

            for iters = 1:num_iters      
                %randomize tst source, new position of random microphone (new
                %position is on circle based off radius_mic), random rotation to
                %microphones, and random sound file (max 4 seconds).
                sourceTest = randSourcePos(1, roomSize, radiusU, ref);
                movingArray = randi(numArrays);
                [~, micsPosNew] = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
           
    %             wav_folder = wavs(3:27);
                rand_wav = randi(25);
                file = wavs(rand_wav+2).name;
                [x_tst,fs_in] = audioread(file);
                [numer, denom] = rat(fs/fs_in);
                x_tst = resample(x_tst,numer,denom);
                x_tst = x_tst';  


              %---- Initialize subnet estimates of training positions ----
                sub_p_hat_ts = zeros(numArrays, 3); 
                for k = 1:numArrays
                    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
                    [~,~,sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                        sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
                end
                [~, p_fail, ~] = moveDetectorOpt(x_tst, transMat, init_var, lambda, eMax, thresh, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                        rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);

        %     %---- estimate test positions after movement ----
                [~,~, p_hat_t] = test(x_tst, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                                numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);

                local_fail = mean(mean(((sourceTest-p_hat_t).^2)));

                if local_fail > modelMean+modelSd        
                    if p_fail > thresh
                        tp_ch_curr = tp_ch_curr + 1;
                    else
                        fn_ch_curr = fn_ch_curr + 1;
                    end                                                       
                end

                if local_fail < modelMean+modelSd
                    if p_fail > thresh
                        fp_ch_curr = fp_ch_curr + 1;
                    else
                        tn_ch_curr = tn_ch_curr + 1;
                    end
                end
            end
        end
        tps(t) = tp_ch_curr/(num_iters*num_radii);
        fps(t) = fp_ch_curr/(num_iters*num_radii);
        tns(t) = tn_ch_curr/(num_iters*num_radii);
        fns(t) = fn_ch_curr/(num_iters*num_radii);
        
        tprs(t) = tps(t)/(tps(t)+fns(t));
        fprs(t) = fps(t)/(fps(t)+tns(t));        
    end
    
    opt_tps = 0;
    opt_fps = 0;
    opt_tns = 0;
    opt_fns = 0;   
    max_auc = 0;    
    for th = 1:num_threshes
        auc_curr = trapz(fprs(t),tprs(t));
        if auc_curr>max_auc
           opt_tps = tps(th);
           opt_fps = fps(th);
           opt_tns = tns(th);
           opt_fns = fns(th);
           max_auc = auc_curr;
        end
    end
end