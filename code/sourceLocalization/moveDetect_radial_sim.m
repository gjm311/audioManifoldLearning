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
load('mat_outputs/monoTestSource_biMicCircle_5L100U.mat')
% load('mat_outputs/movementOptParams')

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
radii = [0:.005:.03];
fs = 32000;
iters_per = 10;
mic_ref = [3 5 1; 5 3 1; 3 1 1; 1 3 1];
accs = struct([]);
wav_folder = dir('./shortSpeech/');

for riter = 1:size(radii)
    acc_curr = struct([]);
    radius_mic = radii(riter);
    
    for it = 1:iters_per 
        %randomize tst source, new position of random microphone (new
        %position is on circle based off radius_mic), random rotation to
        %microphones, and random sound file (max 4 seconds).
        sourceTest = randSourcePos(1, roomSize, radiusU, ref);
        movingArray = randi(numArrays,1);
        randMicPos = randMicsPos(1, roomSize, radius_mic, mic_ref(movingArray,:));
        rotation = rand*360;
        new_x1 = randMicPos(:,1)+.025*cos(rotation);
        new_y1 = randMicPos(:,2)+.025*sin(rotation);
        new_x2 = randMicPos(:,1)-.025*cos(rotation);
        new_y2 = randMicPos(:,2)-.025*sin(rotation);
        if movingArray == 1
            micsPosNew = [new_x1 new_y1 1;new_x2 new_y2 1; micsPos(1+numMics:end,:)];
        elseif movingArray == numArrays
            micsPosNew = [micsPos(1:end-numMics,:); new_x1 new_y1 1; new_x2 new_y2 1];
        elseif movingArray == 2
            micsPosNew = [micsPos(1:numMics,:); new_x1 new_y1 1; new_x2 new_y2 1; micsPos(2*numMics+1:end,:)];
        else
            micsPosNew = [micsPos(1:numMics*2,:); new_x1 new_y1 1; new_x2 new_y2 1; micsPos(3*numMics+1:end,:)];
        end
        rand_wav = randi(numel(wav_folder));
        file = wav_folder(rand_wav).name;
        [x,fs_in] = audioread(file);
        x = resample(x,fs_in,fs);
        x = x';        
        
        %estimate p_fail and estimated positions for each sub_arry
        [self_sub_p_hat_ts, p_fail, posteriors] = moveDetector(x, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
    rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);

        %if p_fail based off MRF is greater than sd of static model we say
        %there is movement.
        if modelSd < p_fail
            acc_curr(it).mrf = 1;
        else
            acc_curr(it).mrf = 0;
        end

        %naive estimate calculated based off standard deviation of new
        %positional estimates and ones taken in training. If estimate sd greater than
        %standard deviation of error of static model then we predict
        %movement.
        naive_p_fail = zeros(1,numArrays);
        for p = 1:numArrays
            %try sd and norm
           naive_p_fail(p) = sd(self_sub_p_hat_ts(p,:)-sub_p_hat_ts(p,:));
           if modelSd < naive_p_fail(p)
                acc_curr(it).naive = 1;
                break
           else
                acc_curr(it).naive = 0;
           end
        end

    end   
    accs(riter).radius = acc_curr;  
end
