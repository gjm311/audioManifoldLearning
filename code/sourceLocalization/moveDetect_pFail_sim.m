
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
load('./mat_results/paramOpt_results_5.mat')
load('mat_outputs/optTransMatData')

%simulate different noise levels
radii = .05:.2:3.05;
num_radii = size(radii,2);
num_lambdas = size(lambdas,2);
num_varis = size(init_vars,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
num_ts = size(T60s,2);
%---- Set MRF params ----
num_iters = 100;
eMax = .3;

p_fails = zeros(num_radii,num_ts,num_iters);

for r = 1:num_radii
    radius_mic = radii(r);
    
    for t = 1:num_ts
        T60 = T60s(t);
        RTF_train = reshape(RTF_trains(t,:,:,:), [nD, rtfLen, numArrays]);    
        scales = scales_t(t,:);
        gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
        p_fail_curr = 0;
        [lam_pos,var_pos] = find(reshape(aucs(:,:,t),[num_lambdas,num_varis]) == max(max(reshape(aucs(:,:,t),[num_lambdas,num_varis]))));
        lambda = lambdas(lam_pos);
        init_var = init_vars(var_pos);
        transMat = reshape(transMats(t,:,:),[3,3]);
        
        
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
            [~, p_fail_out, ~] = moveDetectorOpt(x_tst, transMat, init_var, lambda, eMax, 0, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                    rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            p_fails(r,t,iters) =  p_fail_out;
        end
       
    end
end
    
p_fail = mean(mean(p_fails,2),3);

save('./mat_results/pFail_res_5', 'p_fails', 'p_fail', 'radii')


