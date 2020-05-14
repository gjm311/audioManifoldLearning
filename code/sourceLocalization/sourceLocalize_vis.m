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
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- SIM ROOM SETUP ----

disp('Setting up the room');
load('mat_outputs/monoTestSource_biMicCircle_5L300U_2')

%Chosse T60 from .2, .5. or .8
T60 = .15;
t = find(T60s==T60);
RTF_train = reshape(RTF_trains(t,:,:,:),[nD,rtfLen,numArrays]);
scales = scales_t(t,:);
gammaL = reshape(gammaLs(t,:,:),[nL,nL]);

%Pull random audio file for test
wavs = dir('./shortSpeech/');
rand_wav = randi(25);
file = wavs(rand_wav+2).name;
[x_tst,fs_in] = audioread(file);
[numer, denom] = rat(fs/fs_in);
x_tst = resample(x_tst,numer,denom);
x_tst = x_tst'; 

%---- with optimal params estimate test position ----
% sourceTests = randSourcePos(1, roomSize, radiusU*.35, ref);
% p_hat_ts = zeros(size(sourceTests));
% for i = 1:size(p_hat_ts)
%     sourceTest = sourceTests(i,:);
%     [~,~, p_hat_ts(i,:)] = test(x_tst, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
%                 numMics, sourceTrain, sourceTests(i,:), nL, nU, roomSize, T60, c, fs, kern_typ, scales);
% end

sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
% p_hat_ts = zeros(size(sourceTests));
[~,~, p_hat_ts] = test(x_tst, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
            numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
mse = mean(mean(((sourceTest-p_hat_ts).^2)));
nrm = norm(sourceTest-p_hat_ts);

%---- plot ----
figure(3)
plotRoom(roomSize, micsPos, sourceTrain, sourceTest, nL, p_hat_ts);

title('Source Localization Based Off Semi-Supervised Approach')
grid on
set(gcf,'color','w')
% mseTxt = text(.5,.55, sprintf('MSE (Test Position Estimate): %s(m)', num2str(round(nrm,3))));
ax = gca;
% ax.TitleFontSizeMultiplier  = 2;

% [x_tst,fs_in] = audioread('f0001_us_f0001_00209.wav');
% x_tst = x_tst';
