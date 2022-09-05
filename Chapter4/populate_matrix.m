function K = populate_matrix(k,r1_vals,r2_vals)

n1 = length(r1_vals);
n2 = length(r2_vals);
K = zeros(n1,n2);
for i = 1:n1
    for j = 1:n2
        K(i,j) = k(r1_vals(i),r2_vals(j));
    end
end

end