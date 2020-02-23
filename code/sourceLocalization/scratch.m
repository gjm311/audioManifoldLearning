A = [3 3 3 3; 4 4 4 4; 0 0 0 0 ; 0 0 0 0];

idxs = find(~all(A==0,2));
A = A(idxs,:);
