addpath './mat_outputs'

load('resEsts.mat')

aligns = abs(reshape(align_resids,[1,numArrays*size(align_resids,2)]))./sum(abs(reshape(align_resids,[1,numArrays*size(align_resids,2)])));
misaligns = abs(reshape(misalign_resids,[1,numArrays*size(misalign_resids,2)]))./sum(abs(reshape(misalign_resids,[1,numArrays*size(misalign_resids,2)])));

% edges = zeros(1,1000);
count = 1;
stop = 1;
skip = .0001;
start = skip;
rng = start:skip:stop;
pdfs = zeros(2,stop/skip);

for r = 1:size(pdfs,2)
    mnRng = rng(r)- skip;
    mxRng = rng(r);
    for j = 1:2
        if j == 1
           curr = aligns;
        end
        if j == 2
           curr = misaligns;
        end
    
        for i = 1:size(aligns,2)
            if and(curr(i) < mxRng, curr(i) >= mnRng)
                pdfs(j,r) = pdfs(j,r) + 1;
            end
        end
        pdfs(j,r) = pdfs(j,r)/size(aligns,2); 
    end   
end


figure(1)
bar(pdfs(1,:))
title('Aligned Class [Exponential Distribution]')
xlabel('Residual Error');
ylabel('Probability Density');
xlim([0 100])
ylim([0 max(pdfs(1,:))+.1])

figure(2)
bar(pdfs(2,:))
title('Misaligned Class [Normal Distribution]')
xlim([0 100])
ylim([0 max(pdfs(2,:))+.1])
xlabel('Residual Error');
ylabel('Probability Density');

% figure(1)
% scatter(sortedAligns(:,1), sortedAligns(:,2))
% ylim([0 1])
% figure(2)
% scatter(sortedMisaligns(:,1), sortedMisaligns(:,2))
% ylim([0 1])
% figure(3)
% scatter(sortedUnknowns(:,1), sortedUnknowns(:,2))
% ylim([0 1])
