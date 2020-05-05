
addpath './mat_outputs'
load('resEsts_4.mat')
load('./mat_results/paramOpt_results_4.mat')
num_lambdas = size(lambdas,2);
num_varis = size(init_vars,2);
num_ts = size(T60s,2);

align_resids = align_resids.*100;
misalign_resids = misalign_resids.*100;
% for r = 1:size(align_resids,1)
for r = 1:num_ts
    [bin_aligns,~] = discretize(align_resids(r,:),50);
    [~,bin_edges] = discretize(rmoutliers(align_resids(r,:)),50);
    edges = [al_edges max(misalign_resids(r,:))];
    edges = sort(edges);
    bin_misaligns = discretize(rmoutliers(misalign_resids(r,:)),edges);
    num_bins = 5;
%     all_resids = [align_resids(r,:) misalign_resids(r,:)];
%     [~,edges] = discretize(all_resids,50);
%     bin_aligns = discretize(rmoutliers(align_resids(r,:)),edges);
%     bin_misaligns = discretize(misalign_resids(r,:),edges);
    
    lam_pos = find(mean(reshape(aucs(:,:,r),[num_lambdas,num_varis]),2) == max(mean(reshape(aucs(:,:,r),[num_lambdas,num_varis]),2)));
    var_pos = find(mean(reshape(aucs(:,:,r),[num_lambdas,num_varis]),1) == max(mean(reshape(aucs(:,:,r),[num_lambdas,num_varis]),1)));
    [lam_pos,var_pos] = find(reshape(aucs(:,:,r),[num_lambdas,num_varis]) == max(max(reshape(aucs(:,:,r),[num_lambdas,num_varis]))));
    lambda = lambdas(lam_pos);
    init_var = init_vars(var_pos);
    [avgOpt_lam_pos,avgOpt_var_pos] = find(mean(aucs,3) == max(max(mean(aucs,3))));
    avgOpt_lambda = lambdas(avgOpt_lam_pos);
    avgOpt_init_var = init_vars(avgOpt_var_pos);

    %Generate PDF for NORMAL distribution corresponding to actual data, and
    %based on optimal variance for choice T60 and averaged over all T60s
    figure()
    h = bar(hist(bin_aligns,num_bins)./sum(hist(bin_aligns,num_bins)));
    hold on
    title(sprintf('Empirical Error Distribution: Aligned Network [T60 = %s]\n Avg. SSGP Estimation Error for Aligned Network at Varying Noise Levels', num2str(round(T60s(r),2))))
    az ={round(edges(1:5:end),3)};
    set(gca,'XTickLabel',az)
    ylim([0,1])
    xlabel('Min. SSGP Estimate Error per Bin (m)');
    ylabel('Probability Density');
%     num_bins = size(h.YData,2)+1;
%     opt_pd = makedist('Normal',0,init_var);
%     avgOpt_pd = makedist('Normal',0,avgOpt_init_var);
%     opt_pdf_hist = hist(pdf(opt_pd,sort(align_resids(r,:))),num_bins);
%     avgOpt_pdf_hist = hist(pdf(avgOpt_pd,sort(align_resids(r,:))),num_bins);
%     opt_scale_factor = max(opt_pdf_hist)/(h.YData(1));
%     avgOpt_scale_factor = max(avgOpt_pdf_hist)/(h.YData(1));
%     opt_pdf = opt_pdf_hist./opt_scale_factor;
%     avgOpt_pdf = avgOpt_pdf_hist./avgOpt_scale_factor;
%     pOpt = plot([-1:num_bins],[opt_pdf(1)+.05 opt_pdf(1)+.05 opt_pdf], ':', 'LineWidth',2); 
%     pAvgOpt = plot([-1:num_bins],[avgOpt_pdf(1)+.05 avgOpt_pdf(1)+.05 avgOpt_pdf], '--', 'LineWidth',2);
%     legend([pOpt, pAvgOpt], sprintf('Fitted pdf with optimal variance for specified T60  (%s = %s)','\sigma^2',num2str(round(init_var,2))), sprintf('Fitted pdf based on optimal variance for all T60s (%s = %s)','\sigma^2',num2str(round(avgOpt_init_var,2))))
%     xlim([0,num_bins])
    hold off
    
    %Generate PDFs for EXPONENTIAL distribution corresponding to actual data, and
    %based on optimal variance for choice T60 and averaged over all T60s
    figure()
    h = bar(hist(bin_misaligns,num_bins)./sum(hist(bin_misaligns,num_bins)));
    hold on
    title(sprintf('Empirical Error Distribution: Misaligned Network [T60 = %s]\n Avg. SSGP Estimation Error for Misaligned Network at Varying Noise Levels', num2str(round(T60s(r),2))))
    mz = {round(edges(1:5:end),3)};
    set(gca,'XTickLabel',mz)
    ylim([0,1])
    xlabel('Min. SSGP Estimate Error per Bin (m)');
    ylabel('Probability Density');
%     num_bins = size(h.YData,2);
%     opt_pdf_hist = flip(hist(exppdf(sort(misalign_resids(r,:),'descend'), lambda),num_bins+2));
%     avgOpt_pdf_hist = flip(hist(exppdf(sort(misalign_resids(r,:),'descend'), avgOpt_lambda),num_bins+2));
% %     act_scale_factor = max(act_pdf_hist)/(h.YData(1));
%     opt_scale_factor = max(opt_pdf_hist)/(h.YData(end));
%     avgOpt_scale_factor = max(avgOpt_pdf_hist)/(h.YData(end));
% %     act_pdf = act_pdf_hist./act_scale_factor;
%     opt_pdf = opt_pdf_hist./opt_scale_factor;
%     avgOpt_pdf = avgOpt_pdf_hist./avgOpt_scale_factor;
% %     pAct = plot([-1:num_bins],[act_pdf(1)+.05 act_pdf(1)+.05 act_pdf], 'LineWidth',3);
%     pOpt = plot([-1:num_bins+2],[opt_pdf opt_pdf(end)+1 opt_pdf(end)+1], ':', 'LineWidth',2); 
%     pAvgOpt = plot([-1:num_bins+2],[avgOpt_pdf avgOpt_pdf(end)+1 avgOpt_pdf(end)+1], '--', 'LineWidth',2);
%     legend([pOpt, pAvgOpt], sprintf('Fitted pdf with optimal variance for specified T60  (%s = %s)','\lambda',num2str(round(lambda,2))), sprintf('Fitted pdf based on optimal variance for all T60s (%s = %s)','\lambda',num2str(round(avgOpt_lambda,2))))
% %     legend([pAct, pOpt, pAvgOpt], 'Fitted pdf based on empirical data','Fitted pdf with optimal variance for specified T60', 'Fitted pdf based on optimal variance for all T60s')   
%     xlim([0,num_bins+1])
    hold off
end




% % edges = zeros(1,1000);
% count = 1;
% stop = 1;
% skip = .01;
% start = skip;
% rng = start:skip:stop;
% pdfs = zeros(2,stop/skip);
% 
% for r = 1:size(pdfs,2)
%     mnRng = rng(r)- skip;
%     mxRng = rng(r);
%     for j = 1:2
%         if j == 1
%            curr = aligns;
%            for i = 1:size(aligns,2)
%                 if and(curr(i) < mxRng, curr(i) >= mnRng)
%                     pdfs(j,r) = pdfs(j,r) + 1;
%                 end
%             end
%   
%         end
%         if j == 2
%            curr = misaligns;
%            for i = 1:size(curr,2)
%                 if and(curr(i) < mxRng, curr(i) >= mnRng)
%                     pdfs(j,r) = pdfs(j,r) + 1;
%                 end
%             end
%         end
%     end   
% end

% % aq = 1.5:.45:10.5;
% align_pdfs = pdfs(1,pdfs(1,:)>0); 
% % mq = 1.5:.8:14.5;
% misalign_pdfs = pdfs(2,pdfs(2,:)>0); 
% a_ticks = 0:(1/size(align_pdfs,2)):1;
% m_ticks = 0:(1/size(misalign_pdfs,2)):1;
% % align_pdfs = interp1(align_pdfs,aq);
% % misalign_pdfs = interp1(misalign_pdfs,mq);

% figure(1)
% bar(align_pdfs)
% title('Aligned Residual Error Distribution')
% % xlim([0 20])
% ylim([0 max(pdfs(1,:))+.1])
% set(gca,'XTick',0:size(align_pdfs,2))
% z ={round(a_ticks,2)};
% set(gca,'XTickLabel',z)
% xlabel('Residual Error');
% ylabel('Probability Density');
% ax = gca;
% ax.TitleFontSizeMultiplier  = 2;


% figure(2)
% bar(misalign_pdfs)
% title({'Misaligned Residual Error Distribution'})
% % xlim([0 20])
% ylim([0 max(pdfs(2,:))+.1])
% set(gca,'XTick',0:size(misalign_pdfs,2))
% z = {round(m_ticks,2)};
% set(gca,'XTickLabel',z)
% xlabel('Residual Error');
% ylabel('Probability Density');
% ax = gca;
% ax.TitleFontSizeMultiplier  = 2;