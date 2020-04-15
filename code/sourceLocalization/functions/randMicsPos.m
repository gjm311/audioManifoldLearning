function micPos = randMicsPos(numRand, roomSize, radius, reference)    
    x1 = reference(1);
    y1 = reference(2);
    z1 = reference(3);
    rnd = rand(numRand,1)*pi;
    a=2*rnd*radius;
    r=sqrt(radius);
    micPos = [(radius*r).*cos(a)+x1 (radius*r).*sin(a)+y1 ones(numRand,1)*z1];

end