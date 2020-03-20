function [tot_cost, varis_set] = grad_descent_mrf(radii, cov_init, var_init, max_iters, tol, alpha)
    
    nR = size(radii, 2);
    niter = 1; 
    first = 0;
    
    tot_cost = zeros(1,max_iters+1);
    varis_set = zeros(1,max_iters+1);
    
    %x1: cov, x2: varI, x3: opt. posteriors
    f = @(x1,x2,x3) -.5*x3*inv(x1+x2)*x3' - max(0,.5*log(det(x1+x2))) - .5*nR*log(2*pi);
    tot_cost(1) = -f(cov_init, var_init.*eye(nR), radii);
    varis_set(1) = var_init;

    while and(tot_cost(niter)>=tol, niter <= max_iters)
        if first == 0
            new_vari = var_init;
            new_varI = new_vari*eye(nR);
            new_cov = cov_init;
            first = 1;
        end
        new_gamma = inv(new_cov + new_varI);
        
        dL_dVar = var_grad( radii, new_gamma);
        
        new_vari = new_vari - alpha*dL_dVar;
        new_varI = new_vari*eye(nR);
        
        niter = niter + 1;
        tot_cost(niter) = -f(new_cov, new_varI, radii);
        varis_set(niter) = new_vari;
    end   
end


function dL_dVar = var_grad(radii, gamma)
    dL_dVar = .5*trace((gamma*radii')*(gamma*radii')' - gamma);       
end
