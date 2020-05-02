function [mono_res, sub_res] = gtNaiVary(mono_thresh,sub_thresh,sourceTrain, wavs, gammaL, T60, gt, micRTF_train, micScale, micGammaL, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs)
    num_radii = size(radii, 2);
    nD = size(RTF_train,1);
    mono_tp_ch_curr = 0;
    mono_fp_ch_curr = 0;
    mono_tn_ch_curr = 0;
    mono_fn_ch_curr = 0;
    
    sub_tp_ch_curr = 0;
    sub_fp_ch_curr = 0;
    sub_tn_ch_curr = 0;
    sub_fn_ch_curr = 0;

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
              
          %---- Initialize subnet estimates of training positions ----
            sub_naives = zeros(numArrays,1);
            for k = 1:numArrays
                [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPosNew, RTF_train);
                [~,~,post_sub_p_hat_ts] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
                sub_naives(k) = mean((post_sub_p_hat_ts-sub_p_hat_ts(k,:)).^2);
            end
            
            mono_naives = zeros(numArrays,1);
            naive_p_hat_ts = zeros(numArrays,3);
            for k = 1:numArrays
                subnet = micsPosNew((k-1)*numMics+1:k*numMics,:);
                subscales = micScale(k,:);
                gammaL = reshape(micGammaL(k,:,:),[nL,nL]);
                trRTF = reshape(micRTF_train(k,:,:,:), [nD, rtfLen, numMics]);
                [~,~,naive_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, 1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
%                 sub_naives(k) = mean((naive_p_hat_ts(k,:)-sub_p_hat_ts(k,:)).^2);
            end

            mono_naives(1) = mean(pdist(naive_p_hat_ts(2:4,:)));
            mono_naives(2) = mean(pdist([naive_p_hat_ts(1,:); naive_p_hat_ts(3:4,:)]));
            mono_naives(3) = mean(pdist([naive_p_hat_ts(1:2,:); naive_p_hat_ts(4,:)]));
            mono_naives(4) = mean(pdist(naive_p_hat_ts(1:3,:)));
                 
            if radius_mic >= gt             
                for k = 1:numArrays
                    if mono_naives(k) >= mono_thresh
                        mono_tp_ch_curr = mono_tp_ch_curr+1;
                        break
                    elseif k == 4
                       mono_fn_ch_curr = mono_fn_ch_curr+1;
                    end
                end
                for k = 1:numArrays
                    if sub_naives(k) >= sub_thresh
                        sub_tp_ch_curr = sub_tp_ch_curr+1;
                        break
                    elseif k == 4
                       sub_fn_ch_curr = sub_fn_ch_curr+1;
                    end
                end
            end
            if radius_mic < gt 
                for k = 1:numArrays
                    if mono_naives(k) >= mono_thresh
                        mono_fp_ch_curr = mono_fp_ch_curr+1;
                        break
                    elseif k == 4
                        mono_tn_ch_curr = mono_tn_ch_curr+1;
                    end
                end
                for k = 1:numArrays
                    if sub_naives(k) >= sub_thresh
                        sub_fp_ch_curr = sub_fp_ch_curr+1;
                        break
                    elseif k == 4
                        sub_tn_ch_curr = sub_tn_ch_curr+1;
                    end
                end
                   
            end
        end
    end
    mono_res = [mono_tp_ch_curr mono_fp_ch_curr mono_tn_ch_curr mono_fn_ch_curr];
    sub_res = [sub_tp_ch_curr sub_fp_ch_curr sub_tn_ch_curr sub_fn_ch_curr];
    
end