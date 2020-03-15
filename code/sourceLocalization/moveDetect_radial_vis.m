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

% modelSd = .1;

%simulate different noise levels
radii = 0:0.2:1; 
its_per = 2;
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
accs = struct([]);
wav_folder = dir('./shortSpeech/');
wav_folder = wav_folder(3:27);
inits = 1;
rotation = 0;

transMat = [.75 0.2 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .2;
lambda = .3;
eMax = .3;
thresh = .3;

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
        sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
        if inits > 1
            movingArray = randi(numArrays);
            [rotation, micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
        else
            movingArray = randi(numArrays);
            micsPosNew = micsPos;
            inits = 2;
        end
        
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
        [self_sub_p_hat_ts, p_fail, posteriors] = moveDetectorOpt(x_tst, transMat, init_var, lambda, eMax, thresh, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            
    %---- estimate test positions before movement ----
        [~,~, pre_p_hat_t] = test(x_tst, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);

%     %---- estimate test positions after movement ----
        [~,~, p_hat_t] = test(x_tst, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
        
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
