function [soltranspose] = burgerpde(N,M)

L = 1.00; 
omega=3;
t = linspace(0.0,30.0,N);
x = linspace(0.0,L,M);
m = 0;

sol = pdepe(m,@pdefun,@icfun,@bcfun,x,t);
soltranspose=sol';

writematrix(sol','burgerforcedmatlab1.txt')

plot(x,soltranspose(end-M+1:end),'r')

function [c,f,s] = pdefun(x,t,u,dudx)
c = 1;
f = -(u*u)/2+dudx; 
s = 10*sin(2*pi*x*omega)*cos(t*pi/10)+2*abs(sin(x*pi*omega)*cos(t*pi/10));
end

function u0 = icfun(x)
u0 = 0; 
end

function [pL,qL,pR,qR] = bcfun(xL,uL,xR,uR,t)
pL = uL;
qL = 0;
pR = uR;
qR = 0;
end

end