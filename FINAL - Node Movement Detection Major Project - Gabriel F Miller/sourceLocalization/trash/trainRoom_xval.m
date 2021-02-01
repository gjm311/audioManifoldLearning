%{
This program simulates a room that can be specified in the initialization
parameters with a set of noise sources and microphone nodes. RTFs are
generated (RIRs produced via Emmanuel Habet's RIR generator)and used to
localize sound sources based off a set of learned manifolds (see 
Semi-Supervised Source Localization on Multiple Manifolds With Distributed
Microphones for more details). Gradient descent method used to initialize
hyper-params.
%}
 
% % load desired scenario (scroll past commented section) or uncomment and simulate new environment RTFs 
clear
addpath ./RIR-Generator-master
addpath ./functions
addpath ../../../
load('rtfDatabase.mat')
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech


load('mat_outputs/monoTestSource_biMicCircle_5L300U')
num_iters = 1000;
its_per = 10;
scale_vary = rand(num_iters,size(scales,2));
errs = zeros(1,num_iters);

for it = 1:num_iters
    err_curr = zeros(1,its_per);
    for per = 1:its_per
        scales = scales_set(I,:).*scale_vary(it,:);
        vari = varis_set(I,:);
        % vari = .01;
        [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
        gammaL = inv(sigmaL + diag(ones(1,nL).*vari));
        p_sqL = gammaL*sourceTrainL;

        %---- with optimal params estimate test position ----
        sourceTests = randSourcePos(1, roomSize, radiusU, ref);
        % sourceTests = [2,2,1; 4 4 1];
        p_hat_ts = zeros(size(sourceTests));
        for i = 1:size(p_hat_ts)
            sourceTest = sourceTests(i,:);
            [~,~, p_hat_ts(i,:)] = test(x, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTests(i,:), nL, nU, roomSize, T60, c, fs, kern_typ, scales);
        end
        err_curr(per) = norm(p_hat_ts-sourceTests);
    end
    errs(it) = mean(err_curr);
end

[~,opt_idx] = min(errs);
scales = scales_set(I,:).*scale_vary(opt_idx,:);
save('mat_outputs/monoTestSource_biMicCircle_5L300U');