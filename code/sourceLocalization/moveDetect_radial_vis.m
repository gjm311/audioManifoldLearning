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
load('mat_outputs/monoTestSource_biMicCircle_5L300U')
% load('mat_outputs/movementOptParams')
vari = varis_set(I,:);
% vari = .01;
[~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
gammaL = inv(sigmaL + diag(ones(1,nL).*vari));
p_sqL = gammaL*sourceTrainL;
p_fail = 0;
% modelSd = .1;

%simulate different noise levels
radii = [0 .25 .5];
fs = 32000;
its_per = 2;
N = 5;
mic_ref = [3 5 1; 5 3 1; 3 1 1; 1 3 1];
accs = struct([]);
wav_folder = dir('./shortSpeech/');
wav_folder = wav_folder(3:27);
inits = 1;

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
%         sourceTest = [4 4 1];
        %FIX movingAray for Vis sake
        if inits > 1
            movingArray = 1;
            micsPosNew = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
%             rand_wav = randi(numel(wav_folder));
%             file = wav_folder(rand_wav).name;
%             [x,fs_in] = audioread(file);
%             x = resample(x,fs_in,fs);
%             x = x';  
        else
            micsPosNew = micsPos;
            inits = 2;
        end
        
        p_fail = 0;
        micPos1 = [micLine(micsPos(movingArray,:), micsPosNew(movingArray,:), N), ones(N,1)];
        micPos2 = [micLine(micsPos(movingArray+1,:), micsPosNew(movingArray+1,:), N), ones(N,1)];
        %---- Initialize subnet estimates of training positions ----
        randL = randi(numArrays, 1);
        sub_p_hat_ts = zeros(numArrays, 3); 
        for k = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
            [~,~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
        end
        for t = 1:N
            %estimate p_fail and estimated positions for each sub_array
            %(IN sim must set conditions per rand array)
            micsPosCurr = [micPos1(t,:); micPos2(t,:); micsPos(3:end,:)];
            if t == 1
                [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosCurr, t, p_fail, sub_p_hat_ts, scales, RTF_train,...
        rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            else
                [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosCurr, t, p_fail, self_sub_p_hat_ts, scales, RTF_train,...
        rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            end
        end

        %naive estimate calculated based off standard deviation of new
        %positional estimates and ones taken in training. If estimate sd greater than
        %standard deviation of error of static model then we predict
        %movement.
        naive_p_fail = zeros(1,numArrays);
        for p = 1:numArrays
            %try sd and norm
           naive_p_fail(p) = var(self_sub_p_hat_ts(p,:)-sub_p_hat_ts(p,:));
        end
       

%---- with optimal params estimate test position ----
    p_hat_ts = zeros(size(sourceTest));
    for i = 1:size(p_hat_ts)
        sourceTest = sourceTest(i,:);
        [~,~, p_hat_ts(i,:)] = test(x, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                    numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
    end
    
    %PLOT
    f = plotRoom(roomSize, micsPosNew, sourceTrain, sourceTest, nL, p_hat_ts);
    title('Detecting Array Movement')
    binTxt = text(3.6,.5, sprintf('Movement Detection Probability:\n %s',num2str(round(p_fail,2))));
    naiveTxt = text(.6,.5, sprintf('Avg Subnet Postioional Estimate SD:\n %s',num2str(round(mean(naive_p_fail),2))));
    
%     %Uncomment below for pauses between iterations only continued by
%     %clicking graph  
    w = waitforbuttonpress;
    delete(binTxt);
    delete(naiveTxt);
         
    end
   
end
