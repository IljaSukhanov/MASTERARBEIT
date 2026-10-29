from fractions import Fraction as F
import numpy as np
from numpy.linalg import norm, eigvalsh
from pypdf import PdfReader
from pathlib import Path

# Normalizations in arbitrary embedding dimension from exact sphere moments.
for k in range(2, 21):
    m3, m5 = F(2, k+1), F(8, (k+1)*(k+3))
    l2 = (k*m3-1)/(k-1)
    l4 = ((k+2)*(k+4)*m5-6*(k+2)*m3+3)/((k-1)*(k+1))
    assert l2 == F(1,k+1)
    assert l4 == -F(1,(k+1)*(k+3))
print('PASS: exact degree-two and degree-four cosine multipliers, k=2,...,20')

# Independent planar Fourier coefficients of a translated nonsymmetric triangle.
L = 65536
angles = 2*np.pi*np.arange(L)/L
U = np.column_stack((np.cos(angles), np.sin(angles)))
cloud = np.array([[0.,0.], [2.2,0.3], [-0.4,1.3]])
p = U @ cloud.T
h, he = p.max(axis=1), (p.max(axis=1)-p.min(axis=1))/2
q0 = he.mean()
C = 4*((U.T*he)@U/L-q0*np.eye(2)/2)
a2 = 2*np.mean(h*np.cos(2*angles))
b2 = 2*np.mean(h*np.sin(2*angles))
assert norm(C-np.array([[a2,b2],[b2,-a2]])) < 1e-12
assert abs(q0-h.mean()) < 1e-12
q2 = np.einsum('li,ij,lj->l', U,C,U)
assert abs(np.mean(q2*q2)-norm(C)**2/4) < 1e-12
mu, ck = 2/np.pi, 8/(3*np.pi)
M1 = C/ck+q0/(2*mu)*np.eye(2)
M2 = (3*(U.T*he)@U/L-q0*np.eye(2))/mu
assert norm(M1-M2) < 1e-12
shifted = U @ (cloud+[4,-2]).T
he_shift = (shifted.max(axis=1)-shifted.min(axis=1))/2
assert np.max(abs(he-he_shift)) < 1e-14
residual = he-q0-q2
wmin, wmin2 = 2*he.min(), 2*(q0+eigvalsh(C)[0])
assert abs(wmin-wmin2) <= 2*abs(residual).max()+1e-12
print('PASS: raw/even hull Q0 and Q2, independent Fourier check, M reconstruction, translation and width bound')

# Degree-four pair sum checked against explicit sine/cosine coefficients.
V = U[::128]
H = he[::128]
P4 = 8*(V@V.T)**4-8*(V@V.T)**2+1
pair_energy = 2*np.mean(H[:,None]*H[None,:]*P4)
theta = angles[::128]
a4, b4 = 2*np.mean(H*np.cos(4*theta)), 2*np.mean(H*np.sin(4*theta))
assert abs(pair_energy-(a4*a4+b4*b4)/2) < 1e-12
print('PASS: degree-four directional kernel energy versus explicit Fourier energy')

# Population generator support formulas and AR congruence.
for phi in ([.7], [.7,-.2], [.5,-.2,.1]):
    p = len(phi)
    psi=np.zeros(512)
    psi[0]=1
    for j in range(1,len(psi)):
        psi[j]=sum(phi[r-1]*psi[j-r] for r in range(1,p+1) if j>=r)
    for k in range(p+1,11):
        g=np.array([[psi[j-r] if j>=r else 0 for r in range(k)] for j in range(len(psi))])
        radii=norm(g,axis=1)
        keep=radii>1e-100
        g,radii=g[keep],radii[keep]
        M=(g.T/radii)@g
        B=np.zeros((k-p,k))
        for r in range(k-p):
            B[r,r:r+p+1]=[1,*(-np.array(phi))]
        expected=np.diag(1/radii[:k-p])
        assert norm(B@M@B.T-expected) < 1e-12
        c=np.arange(1,k-p+1)*(-1.)**np.arange(k-p)
        u=B.T@c/norm(B.T@c)
        assert abs(np.sum(abs(g@u))-norm(c,1)/norm(B.T@c)) < 1e-12
print('PASS: AR(1), AR(2), AR(3) filter support and congruence identities through k=10')

pdf=PdfReader(Path(__file__).resolve().parents[3] / 'output/pdf/ar_support_harmonics_summary_v6.pdf')
assert len(pdf.pages)==10
text='\n'.join(page.extract_text() for page in pdf.pages)
for term in ['distributional','unbounded','centroid','stable-law','V-statistic','??']:
    assert term not in text, term
for i,page in enumerate(pdf.pages,1):
    t=page.extract_text()
    assert len(t)>800
    print(f'Page {i}: {len(t)} extracted characters')
print('PASS: ten-page PDF content, no removed topics or unresolved cross-references')
