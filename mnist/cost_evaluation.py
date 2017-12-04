import numpy as np

def comp_dense(Nin, Nout, BA, BW):
    D = Nin+1 #because of bias w^Tx + b is same as [w^T,b]*[x;1]
    return Nout*(D*BA*BW + (D-1)*(BA+BW+np.ceil(np.log2(D))-1))

def rep_dense(Nin, Nout, BA, BW):
    return Nin*BA + (Nin+1)*Nout*BW

print(repr(['Granular Comp', 'ICML Comp', 'Same Comp', 'Granular Rep', 'ICML Rep', 'Same Rep']))

for B in range(1,21):
   comp_granular = comp_dense(784,512,B+2,B+5) + comp_dense(512,512,B+3,B+3) + comp_dense(512,512,B+2,B+2) + comp_dense(512,10,B,B+2)

   comp_icml = comp_dense(784,512,B,B+1) + comp_dense(512,512,B,B+1) + comp_dense(512,512,B,B+1) + comp_dense(512,10,B,B+1)

   comp_same = comp_dense(784,512,B,B) + comp_dense(512,512,B,B) + comp_dense(512,512,B,B) + comp_dense(512,10,B,B)

   rep_granular = rep_dense(784,512,B+2,B+5) + rep_dense(512,512,B+3,B+3) + rep_dense(512,512,B+2,B+2) + rep_dense(512,10,B,B+2)

   rep_icml = rep_dense(784,512,B,B+1) + rep_dense(512,512,B,B+1) + rep_dense(512,512,B,B+1) + rep_dense(512,10,B,B+1)

   rep_same = rep_dense(784,512,B,B) + rep_dense(512,512,B,B) + rep_dense(512,512,B,B) + rep_dense(512,10,B,B)

   print(repr([comp_granular,comp_icml,comp_same,rep_granular,rep_icml,rep_same]))


