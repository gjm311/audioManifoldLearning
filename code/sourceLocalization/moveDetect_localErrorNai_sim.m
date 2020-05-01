
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
load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('mat_outputs/biMicCircle_5L300U_monoNode_4')

%simulate different noise levels
T60s = [0.2 0.4 0.6];
num_ts = size(T60s,2);
radii = 0.05:.2:3.05;
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
num_iters = 100;

monoLocalErrors = zeros(num_radii,num_ts,num_iters);
subLocalErrors = zeros(num_radii,num_ts,num_iters);

for r = 1:num_radii
    radius_mic = radii(r);
    
    for t = 1:num_ts
        T60 = T60s(t);
%         RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
%         scales = scales_t(t,:);
%         gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
        mono_local_error_curr = 0;
        sub_local_error_curr = 0;
        micRTF_train = reshape(micRTF_trains(t,:,:,:), [numArrays, nD, rtfLen, numMics]);
        micScale = reshape(micScales(t,:,:), [numArrays,numMics]);
        micGammaL = reshape(micGammaLs(t,:,:,:), [numArrays,nL,nL]);
        
        for iters = 1:num_iters      
            %randomize tst source, new position of random microphone (new
            %position is on circle based off radius_mic), random rotation to
            %microphones, and random sound file (max 4 seconds).
            sourceTest = randSourcePos(1, roomSize, radiusU, ref);

            movingArray = randi(numArrays);

            [~, micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
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
            post_sub_p_hat_ts = zeros(numArrays, 3); 
            for k = 1:numArrays
                [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
                [~,~,sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
            end
            
            
            %---- Estimate postion after movement via sub-networks ----
            sub_naives = zeros(numArrays,1);
            for k = 1:numArrays
                [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPosNew, RTF_train);
                [~,~,post_sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                    sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
                sub_naives(k) = mean((sub_p_hat_ts(k,:) - post_sub_p_hat_ts(k,:)).^2);
            end
            
                       
            mono_naives = zeros(numArrays,1);
%             sub_naives = zeros(numArrays,1);
            naive_p_hat_ts = zeros(numArrays,3);
            for k = 1:numArrays
        %                 [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
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
            
            monoLocalErrors(r,t,iters) = max(mono_naives);
            subLocalErrors(r,t,iters) = mean(sub_naives);
        end
       
    end
end

monoLocalError = mean(mean(monoLocalErrors,3),2);
subLocalError = mean(mean(subLocalErrors,3),2);


save('./mat_results/localNaiError_res', 'monoLocalError', 'monoLocalErrors', 'subLocalError', 'subLocalErrors', 'radii')
