function [cov,covL,K] = trCovEstDemo(nL, nD, numArrays, rtfs, kernTyp, scales)
    K = zeros(numArrays, nD, nD);
    cov = zeros(1);
    for a = 1:numArrays
        for d1 = 1:nD
            for d2 = 1:nD
                K(a,d1,d2) = kernel(rtfs(d1,:,a), rtfs(d2,:,a), kernTyp, scales(a));
            end
        end
    end
    K = K(:,1:nL,1:nL); 
    covL = zeros(nL,nL);
    for a1 = 1:numArrays
        for a2 = 1:numArrays
            cov = cov + K(a1,:,:).*K(a2,:,:);
        end
    end
    cov = (1/numArrays^2).*cov;

%     cov = zeros(nD,nD);
%     for r = 1:nD
%         for l = 1:nD
%             kern = 0;
%             for i = 1:nD        
%                 for q = 1:numArrays
%                     for w = 1:numArrays
%                         kernQ = kernel(rtfs(r,:,q), rtfs(i,:,q), kernTyp, scales(q));
%                         kernW = kernel(rtfs(l,:,w), rtfs(i,:,w), kernTyp, scales(w));
%                         kern = kern + kernQ*kernW;            
%                     end
%                 end
%             end
%             cov(r,l) = (1/numArrays^2)*kern;
%         end
%     end
    covL = cov(1:nL,1:nL);
end