function k = kernel(sample1,sample2, type, scale)
    if strcmp(type, 'gaussian') == 1
       k = exp(-norm(sample1 - sample2,2)/scale);
    end
end