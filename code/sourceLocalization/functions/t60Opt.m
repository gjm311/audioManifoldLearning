function [mrf_res, sub_res] = t60Opt(nD, thresh, naive_thresh, micScales, micGammaLs, micRTF_train, sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs)
    num_radii = size(radii, 2);
    
    tp_ch_curr = 0;
    fp_ch_curr = 0;
    tn_ch_curr = 0;
    fn_ch_curr = 0;

    sub_tp_ch_curr = 0;
    sub_fp_ch_curr = 0;
    sub_tn_ch_curr = 0;
    sub_fn_ch_curr = 0;

    mean_tp_ch_curr = 0;
    mean_fp_ch_curr = 0;
    mean_tn_ch_curr = 0;
    mean_fn_ch_curr = 0;

    mrfID_corr = 0;
    naiID_corr = 0;
    subNaiID_corr = 0;
    mrfID_uncr = 0;
    naiID_uncr = 0;
    subNaiID_uncr = 0;

    for riter = 1:num_radii
        radius_mic = radii(riter);

        for iters = 1:num_iters      
            %randomize tst source, new position of random microphone (new
            %position is on circle based off radius_mic), random rotation to
            %microphones, and random sound file (max 4 seconds).
            sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
            movingArray = randi(numArrays);
            [~, micsPosNew] = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);

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
            [~, p_fail, posteriors] = moveDetectorOpt(x_tst, transMat, init_var, lambda, eMax, thresh, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                    rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);

        %     %---- estimate test positions after movement ----
            [~,~, p_hat_t] = test(x_tst, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                            numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);

            mean_naives = zeros(numArrays,1);
            sub_naives = zeros(numArrays,1);
            naive_p_hat_ts = zeros(numArrays,3);
            for k = 1:numArrays
        %                 [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
                subnet = micsPos((k-1)*numMics+1:k*numMics,:);
                subscales = micScales(k,:);
                gammaL = reshape(micGammaLs(k,:,:),[nL,nL]);
                trRTF = reshape(micRTF_train(k,:,:,:), [nD, rtfLen, numMics]);
                [~,~,naive_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, 1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
                sub_naives(k) = mean((naive_p_hat_ts(k,:)-sub_p_hat_ts(k,:)).^2);
            end

            mean_naives(1) = mean(pdist(naive_p_hat_ts(2:4,:)));
            mean_naives(2) = mean(pdist([naive_p_hat_ts(1,:); naive_p_hat_ts(3:4,:)]));
            mean_naives(3) = mean(pdist([naive_p_hat_ts(1:2,:); naive_p_hat_ts(4,:)]));
            mean_naives(4) = mean(pdist(naive_p_hat_ts(1:3,:)));

            local_fail = mean(mean(((sourceTest-p_hat_t).^2)));
            mrfMovingArray = find(round(posteriors(:,2),1));

            if local_fail > modelMean+modelSd              
                if movingArray == mrfMovingArray
                    mrfID_corr = mrfID_corr+1;
                else
                    mrfID_uncr = mrfID_uncr+1;
                end

                if p_fail > thresh
                    tp_ch_curr = tp_ch_curr + 1;
                else
                    fn_ch_curr = fn_ch_curr + 1;
                end
                for k = 1:numArrays
                    if sub_naives(k) > naive_thresh
                        sub_tp_ch_curr = sub_tp_ch_curr+1;
                        break
                    elseif k == 4
                       sub_fn_ch_curr = sub_fn_ch_curr+1;
                    end
                end
                if k == movingArray
                   subNaiID_corr = subNaiID_corr + 1;
                else
                   subNaiID_uncr = subNaiID_uncr + 1;
                end

                for k = 1:numArrays
                    if mean_naives(k) > naive_thresh
                        mean_tp_ch_curr = mean_tp_ch_curr+1;
                        break
                    elseif k == 4
                        mean_fn_ch_curr = mean_fn_ch_curr+1;
                    end
                end
                if k == movingArray
                    naiID_corr = naiID_corr+1;
                else 
                    naiID_uncr =  naiID_uncr+1;
                end
            end

            if local_fail < modelMean+modelSd
                if p_fail > thresh
                    fp_ch_curr = fp_ch_curr + 1;
                else
                    tn_ch_curr = tn_ch_curr + 1;
                end
                for k = 1:numArrays
                    if sub_naives(k) > naive_thresh
                        sub_fp_ch_curr = sub_fp_ch_curr+1;
                        break
                    elseif k == 4
                        sub_tn_ch_curr = sub_tn_ch_curr+1;
                    end
                end
                for k = 1:numArrays
                    if mean_naives(k) > naive_thresh
                        mean_fp_ch_curr = mean_fp_ch_curr+1;
                        break
                    elseif k == 4
                        mean_tn_ch_curr = mean_tn_ch_curr+1;
                    end
                end
            end
        end
    end
    mrf_res = [tp_ch_curr fp_ch_curr tn_ch_curr fn_ch_curr]./num_iters*num_radii;
    nai_res = [sub_tp_ch_curr sub_fp_ch_curr sub_tn_ch_curr sub_fn_ch_curr]./num_iters*num_radii;
    sub_res = [mean_tp_ch_curr mean_fp_ch_curr mean_tn_ch_curr mean_fn_ch_curr]./num_iters*num_radii;
    mrfID_res = [mrfID_corr mrfID_uncr]./num_iters*num_radii;
    naiID_res = [naiID_corr naiID_uncr]./num_iters*num_radii;
    subID_res = [subNaiID_corr subNaiID_uncr]./num_iters*num_radii;

end