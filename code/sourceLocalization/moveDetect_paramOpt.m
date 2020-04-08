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
load('mat_outputs/biMicCircle_5L300U_monoNode')
micRTF_train = RTF_train;
micScales = scales;
micVaris = varis;
load('mat_results/vari_t60_data')
load('mat_outputs/monoTestSource_biMicCircle_5L300U')

% load('mat_outputs/movementOptParams')
% vari = varis_set(I,:);
% scales = scales_set(I,:);
[~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales);
gammaL = inv(sigmaL + diag(ones(1,nL).*vari));

%simulate different noise levels
radii = 0:.1:.6;
% num_radii = size(radii,2);
num_radii = 3;
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
T60s = [T60s(1) T60s(3) T60s(6)];
gammas = zeros(3,nL,nL);
gammas(1,:,:) = reshape(gammaLs(1,:,:),[nL, nL]);
gammas(2,:,:) = reshape(gammaLs(3,:,:),[nL, nL]);
gammas(3,:,:) = reshape(gammaLs(6,:,:),[nL, nL]);
gammaLs = gammas;
modelMeans = [modelMeans(1) modelMeans(3) modelMeans(6)];
modelSds = [modelSds(1) modelSds(3) modelSds(6)];
%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_vars = .05:.1:1.05;
lambdas = .05:.1:.75;
eMax = .3;
threshes = 0.3:.1:0.6;
num_threshes = size(threshes,2);
num_iters = 3;
num_ts = size(T60s,2);

numVaris = size(init_vars,2);
numLams = size(lambdas,2);

% local_fails = zeros(num_ts, num_radii, num_threshes);
% p_fails = zeros(num_ts, num_radii, num_threshes);
% tp_check = zeros(1,num_threshes);
% fp_check = zeros(1,num_threshes);
% tn_check = zeros(1,num_threshes);
% fn_check = zeros(1,num_threshes);
wavs = dir('./shortSpeech/');

accs = zeros(numLams, numVaris);
sens = zeros(numLams, numVaris);
specs = zeros(numLams, numVaris);

for lam = 1:numLams
    lambda = lambdas(lam);

    for v = 1:numVaris
        
        init_var = init_vars(v);
        
        tp_ch_curr = 0;
        fp_ch_curr = 0;
        tn_ch_curr = 0;
        fn_ch_curr = 0;
            
        for t = 1:num_ts    
            t_currs = [1 3 6];
            t_curr = t_currs(t);
            T60 = T60s(t);
            modelMean = modelMeans(t);
            modelSd = modelSds(t);
            RTF_train = reshape(RTF_trains(t_curr,:,:,:), [nD, rtfLen, numArrays]);    
            scales = scales_t(t_curr,:);
            gammaL = reshape(gammaLs(t,:,:), [nL, nL]);
            radii = [0 modelMean (modelMean+modelSd)*1.5];

            [tp_out, fp_out, tn_out, fn_out] = paramOpt(sourceTrain, wavs, gammaL, T60, modelMean, modelSd, init_var, lambda, eMax, transMat, RTF_train, nL, nU,rirLen, rtfLen,c, kern_typ, scales, radii,threshes,num_iters, roomSize, radiusU, ref, numArrays, mic_ref, micsPos, numMics, fs);
            tp_ch_curr = tp_ch_curr + tp_out;
            fp_ch_curr = fp_ch_curr + fp_out;
            tn_ch_curr = tn_ch_curr + tn_out;
            fn_ch_curr = fn_ch_curr + fn_out;
        end
        tps = tp_ch_curr./(num_iters*num_ts*num_radii);
        fps = fp_ch_curr./(num_iters*num_ts*num_radii);
        tns = tn_ch_curr./(num_iters*num_ts*num_radii);
        fns = fn_ch_curr./(num_iters*num_ts*num_radii);
%         
%         str(lam,v,1).acc = (tps+tns)/(tps+tns+fns+fps);
%         str(lam,v,2).sens = tps/(tps+fns);
%         str(lam,v,3).spec = tns/(tns+fps);
%        
%         str(lam,v,1) = (tps+tns)/(tps+tns+fns+fps);
%         str(lam,v,2) = tps/(tps+fns);
%         str(lam,v,3) = tns/(tns+fps);
        accs(lam,v) = (tps+tns)/(tps+tns+fns+fps);
        sens(lam,v) = tps/(tps+fns);
        specs(lam,v) = tns/(tns+fps);
    end
    save('paramOpt_results', 'accs', 'sens', 'specs');   
end

