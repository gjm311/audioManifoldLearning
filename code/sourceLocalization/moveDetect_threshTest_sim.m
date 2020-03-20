clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program looks at the localization error (MSE) of arrays when moved 
varying distances from where they originated (i.e. for training). The
purpose is to see the correspondance between the localization error with
the proabilities of failure of the MRF based movement detector.
%}

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
load('mat_outputs/biMicCircle_5L300U_monoNode')
micRTF_train = RTF_train;
micScales = scales;
micVaris = varis;
load('mat_outputs/monoTestSource_biMicCircle_5L300U')
% load('mat_outputs/movementOptParams')
% vari = varis_set(I,:);
% scales = scales_set(I,:);
[~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
gammaL = inv(sigmaL + diag(ones(1,nL).*vari));

%simulate different noise levels
radii = 0:.2:2;
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .1;
lambda = .2;
eMax = .3;
modelMean = .15;
modelSd = .09;
threshes = 0:.01:1;
naive_threshes = .5:.05:5.5;
num_threshes = size(threshes,2);
num_iters = 100;

str = struct([]);

local_fails = zeros(num_radii, num_threshes);
p_fails = zeros(num_radii, num_threshes);
tp_check = zeros(1,num_threshes);
fp_check = zeros(1,num_threshes);
tn_check = zeros(1,num_threshes);
fn_check = zeros(1,num_threshes);

subNai_tp_check = zeros(1,num_threshes);
subNai_fp_check = zeros(1,num_threshes);
subNai_tn_check = zeros(1,num_threshes);
subNai_fn_check = zeros(1,num_threshes);

nai_tp_check = zeros(1,num_threshes);
nai_fp_check = zeros(1,num_threshes);
nai_tn_check = zeros(1,num_threshes);
nai_fn_check = zeros(1,num_threshes);

mrf_micIDs_corr = zeros(1,num_threshes);
subNai_micIDs_corr = zeros(1,num_threshes);
nai_micIDs_corr = zeros(1,num_threshes);
mrf_micIDs_uncr = zeros(1,num_threshes);
subNai_micIDs_uncr = zeros(1,num_threshes);
nai_micIDs_uncr = zeros(1,num_threshes);


for riter = 1:num_radii
    radius_mic = radii(riter);
    
    for thr = 1:num_threshes
        thresh = threshes(thr);
        naive_thresh = naive_threshes(thr);
              
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
        
        for iters = 1:num_iters      
            %randomize tst source, new position of random microphone (new
            %position is on circle based off radius_mic), random rotation to
            %microphones, and random sound file (max 4 seconds).
            sourceTest = randSourcePos(1, roomSize, radiusU, ref);

            movingArray = randi(numArrays);

            [~, micsPosNew] = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
            wavs = dir('./shortSpeech/');
%             wav_folder = wavs(3:27);
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
            [self_sub_p_hat_ts, p_fail, posteriors] = moveDetectorOpt(x_tst, transMat, init_var, lambda, eMax, thresh, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
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
                gammaL = reshape(gammaLs(k,:,:),[nL,nL]);
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
            local_fails(riter, thr) = local_fails(riter, thr)+local_fail;
            p_fails(riter, thr) = p_fails(riter, thr)+p_fail;
            
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
%         p_fails(riter,thr)  = p_fails(riter, thr)./num_iters;
%         str(thr).tp_check = str(thr).tp_check + tp_ch_curr./num_iters;
%         str(thr).fp_check = str(thr).fp_check + fp_ch_curr./num_iters;
%         str(thr).tn_check = str(thr).tn_check + tn_ch_curr./num_iters;
%         str(thr).fn_check = str(thr).fn_check + fn_ch_curr./num_iters;
%         
%         str(thr).subNai_tp_check = str(thr).subNai_tp_check + sub_tp_ch_curr./num_iters;
%         str(thr).subNai_fp_check = str(thr).subNai_fp_check + sub_fp_ch_curr./num_iters;
%         str(thr).subNai_tn_check = str(thr).subNai_tn_check + sub_tn_ch_curr./num_iters;
%         str(thr).subNai_fn_check = str(thr).subNai_fn_check + sub_fn_ch_curr./num_iters;
%         
%         str(thr).nai_tp_check = str(thr).nai_tp_check + mean_tp_ch_curr./num_iters;
%         str(thr).nai_fp_check = str(thr).nai_fp_check + mean_fp_ch_curr./num_iters;
%         str(thr).nai_tn_check = str(thr).nai_tn_check + mean_tn_ch_curr./num_iters;
%         str(thr).nai_fn_check = str(thr).nai_fn_check + mean_fn_ch_curr./num_iters;
%         
%         str(thr).mrf_micIDs_corr = str(thr).mrf_micIDs_corr + mrfID_corr./num_iters;
%         str(thr).subNai_micIDs_corr = str(thr).subNai_micIDs_corr + subNaiID_corr./num_iters;
%         str(thr).nai_micIDs_corr = str(thr).nai_micIDs_corr + naiID_corr./num_iters;
%         
%         str(thr).mrf_micIDs_uncr = str(thr).mrf_micIDs_uncr + mrfID_uncr./num_iters;
%         str(thr).subNai_micIDs_uncr = str(thr).subNai_micIDs_uncr + subNaiID_uncr./num_iters;
%         str(thr).nai_micIDs_uncr = str(thr).nai_micIDs_uncr + naiID_uncr./num_iters;

        p_fails(riter,thr)  = p_fails(riter, thr)./num_iters;
        tp_check(thr) = tp_check(thr) + tp_ch_curr./num_iters;
        fp_check(thr) = fp_check(thr) + fp_ch_curr./num_iters;
        tn_check(thr) = tn_check(thr) + tn_ch_curr./num_iters;
        fn_check(thr) = fn_check(thr) + fn_ch_curr./num_iters;
        
        subNai_tp_check(thr) = subNai_tp_check(thr) + sub_tp_ch_curr./num_iters;
        subNai_fp_check(thr) = subNai_fp_check(thr) + sub_fp_ch_curr./num_iters;
        subNai_tn_check(thr) = subNai_tn_check(thr) + sub_tn_ch_curr./num_iters;
        subNai_fn_check(thr) = subNai_fn_check(thr) + sub_fn_ch_curr./num_iters;
        
        nai_tp_check(thr) = nai_tp_check(thr) + mean_tp_ch_curr./num_iters;
        nai_fp_check(thr) = nai_fp_check(thr) + mean_fp_ch_curr./num_iters;
        nai_tn_check(thr) = nai_tn_check(thr) + mean_tn_ch_curr./num_iters;
        nai_fn_check(thr) = nai_fn_check(thr) + mean_fn_ch_curr./num_iters;
        
        mrf_micIDs_corr(thr) = mrf_micIDs_corr(thr) + mrfID_corr./num_iters;
        subNai_micIDs_corr(thr) = subNai_micIDs_corr(thr) + subNaiID_corr./num_iters;
        nai_micIDs_corr(thr) = nai_micIDs_corr(thr) + naiID_corr./num_iters;
        
        mrf_micIDs_uncr(thr) = mrf_micIDs_uncr(thr) + mrfID_uncr./num_iters;
        subNai_micIDs_uncr(thr) = subNai_micIDs_uncr(thr) + subNaiID_uncr./num_iters;
        nai_micIDs_uncr(thr) = nai_micIDs_uncr(thr) + naiID_uncr./num_iters;

    
    end
    
    save('threshTestResults5', '-v7.3');
end

