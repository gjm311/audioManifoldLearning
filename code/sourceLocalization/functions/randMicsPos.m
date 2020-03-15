function micPos = randMicsPos(numRand, roomSize, radius, reference)
    rand_angle = rand*360;
    
    x1 = reference(1);
    y1 = reference(2);
    z1 = reference(3);
    a=2*pi*rand(numRand,1);
    r=sqrt(rand(numRand,1));
    micPos = [(radius*r).*cos(a)+x1 (radius*r).*sin(a)+y1 ones(numRand,1)*z1];

end