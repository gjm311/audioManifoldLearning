function [p_align, p_misalign]= empProbCheck(sub_error, align_resid,misalign_resid)
    num_splits = 50;
    num_bins = 8;
    [bin_aligns,al_edges] = discretize(rmoutliers((align_resid)),50);
    al_edges = [al_edges max(misalign_resid)];
    al_edges = sort(al_edges);
    [bin_misaligns,mis_edges] = discretize(rmoutliers(misalign_resid,'mean'),al_edges);
    h_al = hist(bin_aligns,num_bins)./sum(hist(bin_aligns,num_bins));
    h_mis = hist(bin_misaligns,num_bins)./sum(hist(bin_misaligns,num_bins));
    h_tot = h_al+h_mis;    
    al_hist = h_al./h_tot;
    al_edges = al_edges(1:ceil(num_splits/num_bins):end);
    mis_edges = mis_edges(1:ceil(num_splits/num_bins):end);
    mis_hist = h_mis./h_tot;
    numEdges = size(al_hist,2);
    
    for edg = 2:numEdges
        if sub_error < al_edges(edg)
            p_align = al_hist(edg-1);
            break
        else
            p_align = al_hist(end);
        end
    end
    for edg = 2:numEdges
        if sub_error < mis_edges(edg)
            p_misalign = mis_hist(edg-1);
        else
            p_misalign = mis_hist(end);
        end
    end
end
