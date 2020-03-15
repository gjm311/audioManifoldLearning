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
radii = 0:.05:.5;
num_radii = size(radii,2);
its_per = 2;
N = 5;
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
accs = struct([]);
wav_folder = dir('./shortSpeech/');
wav_folder = wav_folder(3:27);

%---- Set MRF params ----
transMat = [.75 0.2 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .2;
lambda = .3;
eMax = .3;
thresh = .3;
threshes = 0:.01:1;
num_threshes = size(threshes,2);

local_fails = zeros(num_radii, num_threshes);
p_fails = zeros(num_radii, num_threshes);
thresh_check = zeros(1,num_threshes);

for riter = 1:num_radii
    radius_mic = radii(riter);
    
    for thr = 1:num_threshes
        thresh = threshes(thr);
      
        %randomize tst source, new position of random microphone (new
        %position is on circle based off radius_mic), random rotation to
        %microphones, and random sound file (max 4 seconds).
        sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
        movingArray = randi(numArrays);
        [~, micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
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

%     %---- estimate test positions after movement ----
        [~,~, p_hat_t] = test(x_tst, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
        
        local_fails(riter, thr) = mean(mean(((sourceTest-p_hat_t).^2)));
        p_fails(riter, thr) = p_fail;
        if p_fail > thresh
            thresh_check(thr) = thresh_check(thr) + 1;
        end
    end
   
end

save('./mat_results/threshTestResults')
