function [mrf_res, nai_res, sub_res] =  nodeID(noise,micScales, micGammaLs, micRTF_train, sourceTrain, wavs, gammaL, T60, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs)
    nD = nL+nU;
    num_radii = size(radii, 2);
    thresh = 0;
    mrf_t = 0;
    mrf_f = 0;
    
    mean_t = 0;
    mean_f = 0;
    
    sub_t = 0;
    sub_f = 0;

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
            x_tsts = zeros(numMics*numArrays,size(x_tst,2));
            for nms = 1:numArrays
               x_tsts(numMics*(nms-1)+1,:) = awgn(x_tst,noise);
               x_tsts(numMics*nms,:) = awgn(x_tst,noise);
            end
          %---- Initialize subnet estimates of training positions ----
            sub_p_hat_ts = zeros(numArrays, 3); 
            for k = 1:numArrays
                [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
                [~,~,sub_p_hat_ts(k,:)] = test(x_tsts, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
            end
            [~, ~, posteriors] = moveDetectorOpt(x_tsts, transMat, init_var, lambda, eMax, thresh, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                    rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);

            mean_naives = zeros(numArrays,1);
            sub_naives = zeros(numArrays,1);
            naive_p_hat_ts = zeros(numArrays,3);
            for k = 1:numArrays
        %                 [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
                subnet = micsPos((k-1)*numMics+1:k*numMics,:);
                subscales = micScales(k,:);
                gammaL = reshape(micGammaLs(k,:,:),[nL,nL]);
                trRTF = reshape(micRTF_train(k,:,:,:), [nD, rtfLen, numMics]);
                [~,~,naive_p_hat_ts(k,:)] = test(x_tsts, gammaL, trRTF, subnet, rirLen, rtfLen, 1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
                sub_naives(k) = mean((naive_p_hat_ts(k,:)-sub_p_hat_ts(k,:)).^2);
            end

            mean_naives(1) = mean(pdist(naive_p_hat_ts(2:4,:)));
            mean_naives(2) = mean(pdist([naive_p_hat_ts(1,:); naive_p_hat_ts(3:4,:)]));
            mean_naives(3) = mean(pdist([naive_p_hat_ts(1:2,:); naive_p_hat_ts(4,:)]));
            mean_naives(4) = mean(pdist(naive_p_hat_ts(1:3,:)));
            
            latents = round(posteriors);
            mrf_comp_arr = find(latents(:,1) == 1);
            if mrf_comp_arr == movingArray
                mrf_t = mrf_t+1;
            else
                mrf_f = mrf_f+1;
            end
            
            [~,mean_comp_arr] = min(mean_naives);
            if mean_comp_arr == movingArray
                mean_t = mean_t+1;
            else
                mean_f = mean_f+1;
            end
            
            [~,sub_comp_arr] = max(sub_naives);
            if sub_comp_arr == movingArray
                sub_t = sub_t+1;
            else
                sub_f = sub_f+1;
            end
            
        end
    end
    mrf_res = [mrf_t mrf_f];
    nai_res = [mean_t mean_f];
    sub_res = [sub_t sub_f];
    
end