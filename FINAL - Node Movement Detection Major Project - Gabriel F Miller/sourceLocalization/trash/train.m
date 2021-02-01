%
clear
addpath ./RIR-Generator-master
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% c = 340;
% fs = 8000;
% 
% % ---- TRAINING DATA ----
% % room setup
% disp('Setting up the room');
% 
% roomSize = [6,6,3];    
% % sourceTrainL = [2.9,3,1; 3.1,3,1; 5,5,1; 1,1,1];
% sourceTrainL = [3,3,1];
% sourceTest = [2, 2, 1];
% numArrays = 4;
% numMics = 2;
% radiusL = 2;
% radiusS = .1;
% height = 1;
% nU = 100;
% nL = size(sourceTrainL,1);
% micsPos = [3.1,5,1;2.9,5,1; 5,2.9,1;5,3.1,1; 3.1,1,1;2.9,1,1; 1,2.9,1;1,3.1,1];
% sourceTrainU = randSourcePos(nU,roomSize);
% sourceTrain = [sourceTrainL; sourceTrainU];
% nD = size(sourceTrain,1);
% 
% % ---- generate RIRs and estimate RTFs ----
% totalMics = numMics*numArrays;
% rirLen = 1000;
% rtfLen = 500;
% T60 = 0.15;
% train_sp = randn(size(sourceTrain,1), 10*fs);
% % x = randn(1,10*fs);
% [x,fs_in] = audioread('f0001_us_f0001_00009.wav');
% x = transpose(resample(x,fs,fs_in));
% disp('Generating the train set');
% RTF_train = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
% save('rir_mat/singleSource_biMicCircle_100U')
load('rir_mat/singleSource_biMicCircle_100U')


%---- x-validate to get optimal gauss scale factor and noise estimate ----
scales = 0.01:.01:1;
dBWs = -20:5:35;
scores = zeros(size(scales,2), size(dBWs,2));

for i = 1:size(scales)
    for j = 1:size(dBWs)
        scale = scales(i);
        dBW = dBWs(j);
        
        %---- calculate covariance for labelled nodes ----
        sigmaL = trCovEst(nL, nD, numArrays, RTF_train,'gaussian',scale);
        
        %---- get labelled position values----
        wn = wgn(1,nL,dBW);
        gammaL = inv(sigmaL + diag(ones(1,nL)*var(wn)));
        p_sqL = gammaL*sourceTrainL;
        
        %---- update estimate based off test ----
        %estimate test RTF
        RTF_test =  rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTest, roomSize, T60, rirLen, c, fs);

        %estimate kernel array between labelled data and test
        k_Lt = zeros(1,nL);
        for i = 1:nL
            array_kern = 0;
            for j = 1:numArrays
                k_Lt(i) = array_kern + kernel(RTF_train(i,:,j), RTF_test(:,:,j), 'gaussian', scale);
            end
        end

        %update covariance (same as before but including test as additional unlabelled
        %pt) and update gamma to calculate new gamma.
        RTF_upd = [RTF_train; RTF_test];
        sigmaL_upd = trCovEst(nL, nD+1, numArrays, RTF_upd, 'gaussian',scale);
        gammaL_upd = inv(sigmaL_upd + diag(ones(1,nL)*var(wn)));
        gammaL_new = gammaL_upd + ((gammaL_upd*k_Lt'*k_Lt*gammaL_upd)/(numArrays^2+k_Lt*gammaL_upd*k_Lt'));
        p_sqL_new = gammaL_new*sourceTrainL;

        %estimate test covariance estimate
        sigma_Lt = tstCovEst(nL, nD, numArrays, RTF_upd, RTF_test, 'gaussian', scale);
        sigmaLt_new = sigma_Lt + (1/numArrays)*k_Lt;
        p_hat_t = sigmaLt_new*p_sqL_new;
        
        
        
        score = norm(p_sqL - sourceTrainL,2);
        scores(i,j) = score;       
    end 
end


