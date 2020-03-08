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

%---- Initialize subnet estimates of training positions ----
randL = randi(numArrays, 1);
holderTest = sourceTrain(randL,:);
sub_p_hat_ts = zeros(numArrays, 3); 
for k = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
    [~,~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
        holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
end

%simulate different noise levels
radii = [0 .25 .5];
fs = 32000;
iters_per = 2;
N = 5;
mic_ref = [3 5 1; 5 3 1; 3 1 1; 1 3 1];
accs = struct([]);
wav_folder = dir('./shortSpeech/');
wav_folder = wav_folder(3:27);

for riter = 1:size(radii,2)
    acc_curr = struct([]);
    radius_mic = radii(riter);
    
    for it = 1:iters_per 
        %randomize tst source, new position of random microphone (new
        %position is on circle based off radius_mic), random rotation to
        %microphones, and random sound file (max 4 seconds).
        sourceTests = randSourcePos(1, roomSize, radiusU*.35, ref);
%         sourceTest = [4 4 1];
        %FIX movingAray for Vis sake
        movingArray = 1;
        micsPosNew = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
        rand_wav = randi(numel(wav_folder));
        file = wav_folder(rand_wav).name;
        [x,fs_in] = audioread(file);
        x = resample(x,fs_in,fs);
        x = x';  
        
        t = 20;
        p_fail = 0;
        for t = 1:N
            %estimate p_fail and estimated positions for each sub_array
            [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosNew, t, p_fail, sub_p_hat_ts, scales, RTF_train,...
        rirLen, rtfLen, sourceTrain, sourceTests, nL, nU, roomSize, T60, c, fs, kern_typ);
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

    p_hat_ts = zeros(size(sourceTests));
    for i = 1:size(p_hat_ts)
        sourceTest = sourceTests(i,:);
        [~,~, p_hat_ts(i,:)] = test(x, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                    numMics, sourceTrain, sourceTests(i,:), nL, nU, roomSize, T60, c, fs, kern_typ, scales);
    end
    
    %PLOT
    f = plotRoom(roomSize, micsPosNew, sourceTrain, sourceTest, nL, p_hat_ts);
    title('Detecting Array Movement')
    if p_fail>.7
        binTxt = text(3.6,.5, sprintf('Movement Detection Probability:\n %s (MOVEMENT)',num2str(round(p_fail,2))));
    else
        binTxt = text(3.6,.5, sprintf('Movement Detection Probability:\n %s (NO MOVEMENT)',num2str(round(p_fail,2))));
    end
    if mean(naive_p_fail) > .7
        naiveTxt = text(.6,.5, sprintf('Avg Subnet Postioional Estimate SD:\n %s (MOVEMENT)',num2str(round(mean(naive_p_fail),2))));
    else
        naiveTxt = text(.6,.5, sprintf('Avg Subnet Postioional Estimate SD:\n %s (NO MOVEMENT)',num2str(round(mean(naive_p_fail),2))));
    end
    
%     %Uncomment below for pauses between iterations only continued by
%     %clicking graph  
    w = waitforbuttonpress;
    delete(binTxt);
    delete(naiveTxt);
         
    end
   
end
