function X = helix(theta,R,r,n,w)
    x= (R+r.*cos(n.*theta)).*cos(n.*theta.*w);
    y = (R+r.*cos(n.*theta)).*sin(n.*theta.*w);
    z = r.*sin(n.*theta);
    X = [x,y,z]; 
end