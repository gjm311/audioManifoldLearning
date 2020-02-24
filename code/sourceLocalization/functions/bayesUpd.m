function [mu, cov, Q_t] = bayesUpd(nD, numArrays, RTF_test, scales, k_t_new, p_hat_t, kern_typ, Q_t, mu, cov, vari)
    self_k_new = selfKern(RTF_test,kern_typ,scales,numArrays);
    q_t_new = Q_t*(k_t_new');
    yHat_t_new = q_t_new'*mu;
    norm_new = self_k_new - k_t_new*Q_t*k_t_new';
    h_t_new = cov*q_t_new;
    var_ft_new = norm_new + q_t_new'*h_t_new;
    var_yt_new = vari + var_ft_new;
    
    mu = [mu; yHat_t_new] + (((p_hat_t-yHat_t_new)/(var_yt_new)).*[h_t_new; var_ft_new]);
    Q_t = horzcat(vertcat(Q_t,zeros(1,nD)),zeros(nD+1,1)) + (1/norm_new)*[q_t_new;-1]*[q_t_new;-1]';
    cov = horzcat(vertcat(cov,h_t_new'),[h_t_new;var_ft_new]) - (1/var_yt_new)*[h_t_new;var_ft_new]*[h_t_new;var_ft_new]';
end