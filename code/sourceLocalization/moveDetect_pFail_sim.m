
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
load('mat_outputs/monoTestSource_biMicCircle_5L300U')

%simulate different noise levels
radii = 1.45:.2:3.05;
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
num_iters = 100;

p_fails = zeros(num_radii,num_iters);

for r = 1:num_radii
    radius_mic = radii(r);
    
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
        for k = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
            [~,~,sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
        end
        [~, p_fails(r,iters), ~] = moveDetector(x_tst, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);

    end
end
    
p_fail = mean(p_fails,2);

save('./mat_results/pFail_res2', 'p_fail', 'radii')


