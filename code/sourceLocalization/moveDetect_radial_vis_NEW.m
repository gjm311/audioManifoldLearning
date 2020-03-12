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
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_biMicCircle_5L100U')
% load('mat_outputs/movementOptParams')
vari = varis_set(I,:);
% scales = scales_set(I,:);
% vari = .01;
[~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
gammaL = inv(sigmaL + diag(ones(1,nL).*vari));
% p_sqL = gammaL*sourceTrainL;

% modelSd = .1;

%simulate different noise levels
radii = [0 .025 .05 .075 .1 .3]; 
its_per = 2;
N = 5;
mic_ref = [3 5 1; 5 3 1; 3 1 1; 1 3 1];
accs = struct([]);
wav_folder = dir('./shortSpeech/');
wav_folder = wav_folder(3:27);
inits = 1;
rotation = 0;

for riter = 1:size(radii,2)
    acc_curr = struct([]);
    radius_mic = radii(riter);
    if riter == 1
        iters_per = its_per+1;
    else 
        iters_per = its_per;
    end
    for it = 1:iters_per 
        %randomize tst source, new position of random microphone (new
        %position is on circle based off radius_mic), random rotation to
        %microphones, and random sound file (max 4 seconds).
%         sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
        sourceTest = [4 4 1];
        %FIX movingAray for Vis sake
        if inits > 1
            movingArray = randi(numArrays);
            [rotation, micsPosNew] = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
            rand_wav = randi(numel(wav_folder));
            file = wav_folder(rand_wav).name;
            [x_tst,fs_in] = audioread(file);
            [numer, denom] = rat(fs/fs_in);
            x_tst = resample(x_tst,numer,denom);
            x_tst = x_tst';
        else
            movingArray = 1;
            micsPosNew = micsPos;
            inits = 2;
        end
        
%         p_fail = 0;
%         micPos1 = [micLine(micsPos(movingArray,:), micsPosNew(movingArray,:), N), ones(N,1)];
%         micPos2 = [micLine(micsPos(movingArray+1,:), micsPosNew(movingArray+1,:), N), ones(N,1)];
        %---- Initialize subnet estimates of training positions ----
        sub_p_hat_ts = zeros(numArrays, 3); 
        for k = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
            [~,~,sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
        end
%         for t = 1:N
%             %estimate p_fail and estimated positions for each sub_array
%             %(IN sim must set conditions per rand array)
%             micsPosCurr = [micPos1(t,:); micPos2(t,:); micsPos(3:end,:)];
%             if t == 1 
%                 [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosCurr, t, p_fail, sub_p_hat_ts, scales, RTF_train,...
%         rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
%             else
%                 [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosCurr, t, p_fail, self_sub_p_hat_ts, scales, RTF_train,...
%         rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
%             end
%         end
        [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x_tst, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            
    %---- estimate test positions before movement ----
        [~,~, pre_p_hat_t] = test(x_tst, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
%         pre_p_hat_ts = zeros(size(sourceTest));
%         for i = 1:size(pre_p_hat_ts)
%             [~,~, pre_p_hat_ts(i,:)] = test(x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
%                         numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
%         end
% 
%     %---- estimate test positions after movement ----
        [~,~, p_hat_t] = test(x_tst, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
%         p_hat_ts = zeros(size(sourceTest));
%         for i = 1:size(p_hat_ts)
%             [~,~, p_hat_ts(i,:)] = test(x, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
%                         numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
%         end
        
        %naive estimate calculated based off variance of new
        %positional estimates and ones taken in training. If estimate sd greater than
        %standard deviation of error of static model then we predict
        %movement.
        naive_p_fail = norm(p_hat_t - pre_p_hat_t);
        naive2_p_fail = mean(std(self_sub_p_hat_ts- sub_p_hat_ts));
        %PLOT
        f = plotRoom(roomSize, micsPosNew, sourceTrain, sourceTest, nL, p_hat_t);
        title(sprintf('Detecting Array Movement \n [Array Movement: %s m]', num2str(radius_mic)))
        naiveTxt = text(.6,.5, sprintf('Network Estimate Distance:\n %s',num2str(round(naive_p_fail,2))));
        naive2Txt = text(2.6,.5, sprintf('Std of Sub-Network Estimate Diff.:\n %s',num2str(round(naive2_p_fail,2))));
        binTxt = text(4.6,.5, sprintf('MRF Based Probability:\n %s',num2str(round(p_fail,2))));
        %     %Uncomment below for pauses between iterations only continued by
    %     %clicking graph  
        w = waitforbuttonpress;
        delete(binTxt);
        delete(naiveTxt);
        delete(naive2Txt);
        clf
         
    end
   
end
