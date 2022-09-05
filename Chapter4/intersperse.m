% Construct a set of centres that do not lie near data points
% factor controls the number of points
function out = intersperse(rvals,factor)

if size(rvals,1) == 1; rvals = rvals'; end % want a column vector

range = max(rvals) - min(rvals);

rvals = sort(rvals);
rvals = [min(rvals) - 0.1*range; ...
         rvals; ...
         max(rvals) + 0.1*range];
drvals = diff(rvals); 

out = [];
for i = 1:factor
    out = [out; ...
                 rvals(1:end-1) + (i / (factor + 1)) * drvals];
end
out = sort(out);

end