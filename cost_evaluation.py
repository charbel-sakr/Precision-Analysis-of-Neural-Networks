import numpy as np

def comp_convolution(Nin, Nout, kw, kout, BA, BW):
    DPs = Nout*kout*kout
    D = Nin*kw*kw
    return comp_dense(D,DPs,BA,BW)

def rep_convolution(Nin, kin, Nout, kw, BA,BW):
    Nw = Nin*Nout*kw*kw + Nout #kernels and bias per output channel
    return BW*Nw + BA*Nin*kin*kin

def comp_dense(Nin, Nout, BA, BW):
    D = Nin+1 #because of bias w^Tx + b is same as [w^T,b]*[x;1]
    return Nout*(D*BA*BW + (D-1)*(BA+BW+np.ceil(np.log2(D))-1))

def rep_dense(Nin, Nout, BA, BW):
    return Nin*BA + (Nin+1)*Nout*BW

print(repr(['Granular Comp', 'ICML Comp', 'Same Comp', 'Granular Rep', 'ICML Rep', 'Same Rep']))

for B in range(1,21):
   comp_granular = comp_convolution(3,32,3,30,B+6,B+8)
   comp_granular += comp_convolution(32,32,3,28,B+2,B+8)
   comp_granular += comp_convolution(32,64,3,12,B+3,B+9)
   comp_granular += comp_convolution(64,64,3,10,B+2,B+8)
   comp_granular += comp_convolution(64,128,3,3,B+3,B+7)
   comp_granular += comp_convolution(128,128,3,1,B+2,B+6)
   comp_granular += comp_dense(128,256,B+3,B+4)
   comp_granular += comp_dense(256,256,B+1,B+4)
   comp_granular += comp_dense(256,10,B+3,B)

   comp_icml = comp_convolution(3,32,3,30,B,B+4)
   comp_icml += comp_convolution(32,32,3,28,B,B+4)
   comp_icml += comp_convolution(32,64,3,12,B,B+4)
   comp_icml += comp_convolution(64,64,3,10,B,B+4)
   comp_icml += comp_convolution(64,128,3,3,B,B+4)
   comp_icml += comp_convolution(128,128,3,1,B,B+4)
   comp_icml += comp_dense(128,256,B,B+4)
   comp_icml += comp_dense(256,256,B,B+4)
   comp_icml += comp_dense(256,10,B,B+4)

   comp_same =comp_convolution(3,32,3,30,B,B)
   comp_same += comp_convolution(32,32,3,28,B,B)
   comp_same += comp_convolution(32,64,3,12,B,B)
   comp_same += comp_convolution(64,64,3,10,B,B)
   comp_same += comp_convolution(64,128,3,3,B,B)
   comp_same += comp_convolution(128,128,3,1,B,B)
   comp_same += comp_dense(128,256,B,B)
   comp_same +=comp_dense(256,256,B,B)
   comp_same += comp_dense(256,10,B,B)

   rep_granular = rep_convolution(3,32,32,3,B+6,B+8)
   rep_granular += rep_convolution(32,30,32,3,B+2,B+8)
   rep_granular += rep_convolution(32,28,64,3,B+3,B+9)
   rep_granular += rep_convolution(64,12,64,3,B+2,B+8)
   rep_granular += rep_convolution(64,10,128,3,B+3,B+7)
   rep_granular += rep_convolution(128,3,128,3,B+2,B+6)
   rep_granular += rep_dense(128,256,B+3,B+4)
   rep_granular += rep_dense(256,256,B+1,B+4)
   rep_granular += rep_dense(256,10,B+3,B)

   rep_icml = rep_convolution(3,32,32,3,B,B+4)
   rep_icml += rep_convolution(32,30,32,3,B,B+4)
   rep_icml += rep_convolution(32,28,64,3,B,B+4)
   rep_icml += rep_convolution(64,12,64,3,B,B+4)
   rep_icml += rep_convolution(64,10,128,3,B,B+4)
   rep_icml += rep_convolution(128,3,128,3,B,B+4)
   rep_icml += rep_dense(128,256,B,B+4)
   rep_icml += rep_dense(256,256,B,B+4)
   rep_icml += rep_dense(256,10,B,B+4)

   rep_same = rep_convolution(3,32,32,3,B,B)
   rep_same += rep_convolution(32,30,32,3,B,B)
   rep_same += rep_convolution(32,28,64,3,B,B)
   rep_same += rep_convolution(64,12,64,3,B,B)
   rep_same += rep_convolution(64,10,128,3,B,B)
   rep_same += rep_convolution(128,3,128,3,B,B)
   rep_same += rep_dense(128,256,B,B)
   rep_same += rep_dense(256,256,B,B)
   rep_same += rep_dense(256,10,B,B)

   print(repr([comp_granular,comp_icml,comp_same,rep_granular,rep_icml,rep_same]))


