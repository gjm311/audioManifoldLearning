clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% load('./mat_results/threshTestResults4')

load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('./mat_results/paramOpt_results_5.mat')

radii = [0 .65 .85 1.5];
% num_radii = size(radii,2);
num_radii = size(radii,2);
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];

%---- Set MRF params ----
transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];

eMax = .3;
threshes = 0:.25:1;
num_threshes = size(threshes,2);
num_iters = 5;
numVaris = size(init_vars,2);
numLams = size(lambdas,2);
num_ts = size(T60s,2);


for t=1:num_ts+1
    figure(t)
    grid on
    set(gcf,'color','w')
    if t==num_ts+1
        h = heatmap(mean(aucs,3));
        title(sprintf('AUCs for Varying MRF Hyper-Parameters \n Averaged for all T60s')) 
    else
        h = heatmap(reshape(aucs(:,:,t),[numLams,numVaris]));
        title(sprintf('AUCs for Varying MRF Hyper-Parameters \nT60 = %s',num2str(round(T60s(t),2))))
    end
    %     title(sprintf('AUCs for Varying MRF Hyper-Parameters \n Averaged for all T-60s'))
    ylabel('\sigma^2')
    xlabel('\lambda')
    % ax = gca;
    h.XData = round(lambdas,2);
    h.YData = round(init_vars,2);
    ax = gca;
end
grid on

