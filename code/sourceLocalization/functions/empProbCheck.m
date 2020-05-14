function [p_align, p_misalign]= empProbCheck(sub_error, align_resid,misalign_resid)
    num_splits = 50;
    [bin_aligns,~] = discretize(align_resid,num_splits);
    [~, al_edges] = discretize(rmoutliers(align_resid),num_splits);
    edges = [al_edges max(misalign_resid)];
    edges = sort(edges);
    bin_misaligns = discretize(misalign_resid,edges);
    num_bins = 5;
    h_al = hist(bin_aligns,num_bins)./sum(hist(bin_aligns,num_bins));
    h_mis = hist(bin_misaligns,num_bins)./sum(hist(bin_misaligns,num_bins));
    h_tot = h_al+h_mis;    
    al_hist = h_al./h_tot;
    edges = edges(1:num_splits/num_bins:end);
    mis_hist = h_mis./h_tot;
    numEdges = size(al_hist,2);
    
    for edg = 2:numEdges
        if sub_error < edges(edg)
            p_align = al_hist(edg-1);
            break
        else
            p_align = al_hist(end);
        end
    end
    for edg = 2:numEdges
        if sub_error < edges(edg)
            p_misalign = mis_hist(edg-1);
        else
            p_misalign = mis_hist(end);
        end
    end
end
