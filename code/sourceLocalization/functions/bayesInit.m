function [mu, cov, Q_t] = bayesInit(nD, sourceTrain, RTF_train, kern_typ, scales, numArrays, vari)
    init_selfK = selfKern(RTF_train(1,:,:),kern_typ,scales,numArrays);
    mu = (sourceTrain(1,:)*init_selfK)/(vari+init_selfK);
    cov = init_selfK - (init_selfK^2/(vari + init_selfK));
    Q_t = 1/init_selfK;
    
    for ii = 2:nD
        RTF_test = RTF_train(ii,:,:);
        k_Lt = zeros(1,ii-1);
        for i = 1:ii-1
            for j = 1:numArrays
                k_Lt(i) = k_Lt(i) + kernel(RTF_train(i,:,j), RTF_test(:,:,j), kern_typ, scales(j));
            end
        end
        
        
%         k_Lt = zeros(1,ii-1);
%         for i = 1:ii-1
%             array_kern = 0;
%             for j = 1:numArrays
%                 k_Lt(i) = array_kern + kernel(RTF_train(i,:,j), RTF_train(ii,:,j), kern_typ, scales(j));
%             end
%         end
        [mu, cov, Q_t] = bayesUpd(ii-1, numArrays, RTF_train(ii,:,:), scales, sourceTrain(ii,:), kern_typ, k_Lt, Q_t, mu, cov, vari);
    end
end