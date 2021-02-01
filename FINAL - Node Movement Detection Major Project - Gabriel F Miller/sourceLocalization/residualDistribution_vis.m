addpath './mat_outputs'

%{
This program visualizes the residual error for a series of different T60s
and noise levels. This is done for both aligned arrays (i.e. in the same 
position that was used when the ground truth was measured) and misaligned
(random array moved to random position in room). Distributions generated 
inform the choices for the latent state prior distributions of the MRF.
%}

load('resEsts_4.mat')
load('./mat_results/paramOpt_results_5.mat')
num_lambdas = size(lambdas,2);
num_varis = size(init_vars,2);
num_ts = size(T60s,2);

for r = 1:num_ts
    [bin_aligns,al_edges] = discretize(rmoutliers(align_resids(r,:)),50);
    [~,bin_edges] = discretize(rmoutliers(align_resids(r,:)),50);
    edges = [al_edges max(misalign_resids(r,:))];
    edges = sort(edges);
    [bin_misaligns,mis_edges] = discretize(rmoutliers(misalign_resids(r,:),'mean'),50);
    num_bins = 8;

    [lam_pos,var_pos] = find(reshape(aucs(:,:,r),[num_lambdas,num_varis]) == max(max(reshape(aucs(:,:,r),[num_lambdas,num_varis]))));
    lambda = lambdas(lam_pos(1));
    init_var = init_vars(var_pos(1));
    [avgOpt_lam_pos,avgOpt_var_pos] = find(mean(aucs,3) == max(max(mean(aucs,3))));
    avgOpt_lambda = lambdas(avgOpt_lam_pos);
    avgOpt_init_var = init_vars(avgOpt_var_pos);

    %Generate PDF for NORMAL distribution corresponding to actual data, and
    %based on optimal variance for choice T60 and averaged over all T60s
    figure()
    grid on
    set(gcf,'color','w')
%     h_al = hist(bin_aligns,num_bins)./sum(hist(bin_aligns,num_bins));
%     h_mis = hist(bin_misaligns,num_bins)./sum(hist(bin_misaligns,num_bins));
%     h_tot = h_al+h_mis;    
%     h = bar(h_al./h_tot);

    h = bar(hist(bin_aligns,num_bins)./sum(hist(bin_aligns,num_bins)));
    hold on
    title(sprintf('Empirical Error Distribution: Aligned Network [T60 = %s]\n Avg. SSGP Estimation Error for Aligned Network at Varying Noise Levels', num2str(round(T60s(r),2))))
%     title(sprintf('Relative Empirical Error Distribution: Aligned Network [T60 = %s]\n Relative SSGP Estimation Error for Misaligned Network at Varying Noise Levels', num2str(round(T60s(r),2))))
    az ={round(al_edges(1:5:end),3)};
    set(gca,'XTickLabel',az)
    ylim([0,1])
    xlabel('Min. SSGP Estimate Error per Bin (m)');
    ylabel('Estimated Density');
%     num_bins = num_bins+1;
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
    grid on
    
    %Generate PDFs for EXPONENTIAL distribution corresponding to actual data, and
    %based on optimal variance for choice T60 and averaged over all T60s
    figure()
    grid on
    set(gcf,'color','w')
%     h = bar(h_mis./h_tot);
    h = bar(hist(bin_misaligns,num_bins)./sum(hist(bin_misaligns,num_bins)));
    hold on
    title(sprintf('Empirical Error Distribution: Misaligned Network [T60 = %s]\n Avg. SSGP Estimation Error for Misaligned Network at Varying Noise Levels', num2str(round(T60s(r),2))))
%     title(sprintf('Relative Empirical Error Distribution: Misaligned Network [T60 = %s]\n Relative SSGP Estimation Error for Misaligned Network at Varying Noise Levels', num2str(round(T60s(r),2))))
    mz = {round(mis_edges(1:5:end),3)};
    set(gca,'XTickLabel',mz)
    ylim([0,1])
    xlabel('Min. SSGP Estimate Error per Bin (m)');
    ylabel('Estimated Density');
%     num_bins = num_bins+1;
%     opt_pdf_hist = hist(exppdf(sort(misalign_resids(r,:)), lambda),num_bins);
%     avgOpt_pdf_hist = hist(exppdf(sort(misalign_resids(r,:)), avgOpt_lambda),num_bins);
% %     act_scale_factor = max(act_pdf_hist)/(h.YData(1));
%     opt_scale_factor = max(opt_pdf_hist)/(h.YData(1));
%     avgOpt_scale_factor = max(avgOpt_pdf_hist)/(h.YData(1));
% %     act_pdf = act_pdf_hist./act_scale_factor;
%     opt_pdf = opt_pdf_hist./opt_scale_factor;
%     avgOpt_pdf = avgOpt_pdf_hist./avgOpt_scale_factor;
% %     pAct = plot([-1:num_bins],[act_pdf(1)+.05 act_pdf(1)+.05 act_pdf], 'LineWidth',3);
%     pOpt = plot([-1:num_bins],[opt_pdf(1)+.05 opt_pdf(1)+.05 opt_pdf], ':', 'LineWidth',2); 
%     pAvgOpt = plot([-1:num_bins],[avgOpt_pdf(1)+.05 avgOpt_pdf(1)+.05 avgOpt_pdf], '--', 'LineWidth',2);
%     legend([pOpt, pAvgOpt], sprintf('Fitted pdf with optimal variance for specified T60  (%s = %s)','\lambda',num2str(round(lambda,2))), sprintf('Fitted pdf based on optimal variance for all T60s (%s = %s)','\lambda',num2str(round(avgOpt_lambda,2))))
% %     legend([pAct, pOpt, pAvgOpt], 'Fitted pdf based on empirical data','Fitted pdf with optimal variance for specified T60', 'Fitted pdf based on optimal variance for all T60s')   
%     xlim([0,num_bins])
    hold off
    grid on
end