function [H1,H2, LapN, D] = SpectralClusteringqi(CKSym, c)
warning off;
DN = diag( 1./sqrt(sum(CKSym)+eps) );
LapN = DN * CKSym * DN;
[H1] = eig2(LapN, c);
D = diag( 1./sqrt(diag(H1*H1')) );
H2 = H1 ./ sqrt(sum(H1.^2, 2));
if any(isnan(H2))
    H2(isnan(H2)) = 0;
end
H2 = real(H2);
end