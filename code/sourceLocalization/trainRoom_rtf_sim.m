%{
This program simulates a room with parameters that can be specified. 
Available options for sound source and microphone node positioning as well. 
RTFs are generated (RIRs produced via Emmanuel Habet's RIR generator)and used to
localize sound sources based off a set of learned manifolds (see 
Semi-Supervised Source Localization on Multiple Manifolds With Distributed
Microphones for more details). Gradient descent method used to initialize
hyper-params.
%}
 
clear
addpath ./RIR-Generator-master
addpath ./functions
addpath ../../../
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
% ---- Initialize Parameters for training ----
c = 340;
fs = 8000;
roomSize = [6,6,3]; 
ref = [roomSize(1:2)./2 1];

% Labelled training point coordinates if grid desired.
% height = 1.5;
% width = 1.5;
% numRow = 7;
% numCol = 7;
% sourceTrainL = sourceGrid(height, width, numRow, numCol, ref);

%Labelled training point coordinates
sourceTrainL = [4,4,1; 2,2,1; 4,2,1; 2,4,1; 3 3 1];

micsPos = [3.025,5.75 1;2.975,5.75,1; 5.75,2.975,1;5.75,3.025,1; 3.025,.25,1;2.975,.25,1; .25,2.975,1;.25,3.025,1];   %mics position (spaced 5 cm apart)
numArrays = 4;   %number of arrays
numMics = 2;   %mics per array
nU = 300;   %unlabelled points
radiusU = 2;   %radius around center of room which unlabelled data is distributed.
nL = size(sourceTrainL,1);
sourceTrainU = randSourcePos(nU, roomSize, radiusU, ref); %generate random unlabelled positions within radiusU of center of the room (i.e. ref)
sourceTrain = [sourceTrainL; sourceTrainU];
nD = size(sourceTrain,1);

totalMics = numMics*numArrays;
rirLen = 1000;
rtfLen = 500;
T60s = [.2 .5 .8];
num_ts = size(T60s,2);

%grad descent parametrs for optimizing SSGP params
x = randn(1,10*fs);
kern_typ = 'gaussian';
tol = 10e-3;
alpha = .01;
max_iters = 100;
init_scales = 1*ones(1,numArrays);
var_init = rand(1,nL)*.1-.05;

RTF_trains = zeros(num_ts, nD, rtfLen, numArrays);
scales_t = zeros(num_ts, numArrays);
gammaLs = zeros(num_ts, nL, nL);

disp('Generating the train set');

%Iterate through different t-60s and optimize SSGP parameters per t-60
for t = 1:num_ts
    T60 = T60s(t);
    
    % ---- Simulate RTFs between nodes ----
    RTF_train_curr = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
    RTF_trains(t,:,:,:) = RTF_train_curr;
    
    % ---- load training data (check mat_trainParams for options)----

    %Initialize hyper-parameter estimates (gaussian kernel scalar
    %value and noise variance estimate)
    [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train_curr, kern_typ, init_scales);

    %---- perform grad. descent to get optimal params ----
    [costs, ~, ~, varis_set, scales_set] = grad_descent(sourceTrainL, numArrays, RTF_train_curr, sigmaL, init_scales, var_init, max_iters, tol, alpha, kern_typ);
    [~,I] = min(sum(costs));

    scales_t(t,:) = scales_set(I,:);
    vari = varis_set(I,:);
    [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train_curr, kern_typ, scales_t(t,:));
    gammaLs(t,:,:) = inv(sigmaL + diag(ones(1,nL).*vari));
end

%Change number so not to overwrite previous models
save('mat_outputs/monoTestSource_biMicCircle_5L300U_6')