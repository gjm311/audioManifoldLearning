
clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
%---- load training data (check mat_trainParams for options)----
load('mat_results/vari_per_t60.mat')
load('mat_outputs/monoTestSource_biMicCircle_5L300U.mat')
%simulate different noise levels
radii = 0:.2:2;
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
wav_folder = dir('./shortSpeech/');
wav_folder = wav_folder(3:27);

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .2;
lambda = .2;
eMax = .3;
thresh = .3;
threshes = 0:.01:1;
num_threshes = size(threshes,2);
num_iters = 100;
num_t60s = size(T60s,2);

resids = zeros(num_radii,num_iters,numArrays);
theta1_opts = zeros(1,num_t60s);
theta2_opts = zeros(1,num_t60s);
mis_posteriors = zeros(num_iters,num_radii);

for t = 1:num_t60s
    if t > 1
       init_var = theta1_opts(t-1);
       lambda = theta2_opts(t-1);
    end
    T60 = T60s(t);
    RTF_train = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
    [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, init_scales);

    %---- perform grad. descent to get optimal params ----
    gammaL = inv(sigmaL + diag(ones(1,nL).*var_init));
    p_sqL = gammaL*sourceTrainL;

    [costs, ~, ~, varis_set, scales_set] = grad_descent(sourceTrainL, numArrays, RTF_train, sigmaL, init_scales, var_init, max_iters, tol, alpha, kern_typ);
    [~,I] = min(sum(costs));
    scales = scales_set(I,:);
    vari = varis_set(I,:);
    [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
    gammaL = inv(sigmaL + diag(ones(1,nL).*vari));
    
    theta1s = zeros(num_iters,num_radii);
    theta2s = zeros(num_iters,num_radii);
    
    for riter = 1:num_radii
        radius_mic = radii(riter);

        for iter = 1:num_iters
            %randomize tst source, new position of random microphone (new
            %position is on circle based off radius_mic), random rotation to
            %microphones, and random sound file (max 4 seconds).
            sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
            movingArray = randi(numArrays);
            [~, micsPosNew] = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);

            rand_wav = randi(numel(wav_folder));
            file = wav_folder(rand_wav).name;
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

            %---- Initialize CRF params ----
            posteriors = zeros(numArrays,3);
            likes = zeros(numArrays,3);
            latents = [ones(numArrays,1), zeros(numArrays,1), zeros(numArrays,1)]+1;

            % ----Calculate new estimate based off movement for all subnets and resid. error from stationary time (turns off once estimates settle) ----
            upd_sub_p_hat_ts = zeros(numArrays, 3);
            for k1 = 1:numArrays
                [subnet, subscales, trRTF] = subNet(k1, numArrays, numMics, scales, micsPosNew, RTF_train);
                [~,~,upd_sub_p_hat_ts(k1,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);   
            end
            resids(riter,iter,:) = mean(sub_p_hat_ts-upd_sub_p_hat_ts,2);


            %---- Set likelihoods (to be maximized during msg passing calc.) ----
            for k = 1:numArrays
                theta1 = 2*normpdf(resids(k)*10, 0, init_var);
                theta1s(riter,iter) = theta1;
                theta2 = (lambda*exp(-lambda*resids(k)))/(1-exp(-lambda*eMax));
                theta2s(riter,iter) = theta2;
                theta3 = unifrnd(0,eMax);
                likes(k,:) = [theta1 theta2 theta3];
            end

            % ---- For each incoming sample, use CRF to determine prob of latents and detection failure ----
            %Assuming fully connected network, we begin by allowing each latent
            %variable to receive messages from its connected variables to
            %initialize marginal posteriors.
            mu_alphas = zeros(numArrays,3);
            for k1 = 1:numArrays
                for l = 1:3
                    for k2 = 1:numArrays
                        if k2 ~= k1
                            mu_alphas(k1,l) = transMat(latents(k2,l), latents(k1,l))*(likes(k2,l)); 
                        end
                    end
                    posteriors(k1,l) = likes(k1,l)*mu_alphas(k1,l); 
                end
                posteriors(k1,:) = posteriors(k1,:)./(sum(posteriors(k1,:)));
            end

            %Variables send further messages using arbitrary passing strategy.
            mu_alpha_prev = zeros(1,3);
            tol = 10e-3;
            err = inf;
            while err > tol
                if err == inf
                    prev_posts = zeros(size(posteriors));
                end
                for k = 1:numArrays
                    for l = 1:3
                        if k == 1
                            mu_alpha = transMat(latents(1,l), latents(2,l))*(likes(k,l));
                            mu_alpha_prev(l) = mu_alpha;
                        end
                        if k == numArrays
                            mu_alpha = transMat(latents(k-1,l), latents(k,l))*mu_alpha_prev(l);
                            mu_alpha_prev(l) = mu_alpha;
                        end 
                        if and(k>1,k<numArrays)
                            mu_alpha = transMat(latents(k-1,l), latents(k,l))*mu_alpha_prev(l);
                            mu_alpha_prev(l) = mu_alpha;
                        end
                        posteriors(k,l) = likes(k,l)*mu_alpha;
                    end
                    posteriors(k,:) = posteriors(k,:)./(sum(posteriors(k,:)));
                end
                err = norm((posteriors-prev_posts), 2);
                prev_posts = posteriors;
            end

            latents = round(posteriors);
            MIS = sum(latents(:,2))/(numArrays - sum(latents(:,3))+10e-6);
            if MIS > thresh
                p_fail = 1;
            else
                p_fail = 0;
            end

            mis_probs = posteriors(:,2);
            t = 1;
            p_fails = 0;
            if sum(latents(:,2)) > 0
                mis_posteriors(riter,iter) = (1/t)*(p_fails*(t-1)+p_fail)*(mean(mis_probs(latents(:,2) == 1)));
            else
                mis_posteriors(riter,iter) = (1/t)*(p_fails*(t-1)+p_fail);
            end
        end
    end

    mean_resids = reshape(mean(resids,2),[num_radii,numArrays]);
    resids_cov = zeros(num_radii, num_radii);
    for r1 = 1:num_radii
        for r2 = 1:num_radii
           resids_cov(r1,r2) = kernel(mean_resids(r1,:),mean_resids(r2,:),kern_typ,1);  
        end    
    end

    
    
    theta1_mean = sum(sum(theta1s))./(num_iters*num_radii);
    theta1_opts(t) = sum(sum((theta1s-theta1_mean).^2))./(num_iters*num_radii);
    theta2_opts(t) = sum(sum(theta2s))./(num_iters*num_radii);
end

save('mat_results/movementOptParams')