# Precision-Analysis-of-Neural-Networks
Code needed to generate results from my ICML 2017 and ICASSP 2018 papers.

Here you will find theano code needed to do the following: Training of baseline neural network, removal of dropout and batchnorm layers to obtain clean pre-trained networks (only dot products and activations), quantization noise gains calculation (the E values), inference in fixed-point (fixed-point simulation), and computational and representational costs evaluation routines (in order to make it easier to compute these costs given a network architecture), and all needed helper files.

You won't find the datasets, please download these on your own. I also haven't uploaded the models (they're too big) but the scripts should make it easy for you to build them.

Please get in touch if you have any question or comment.

Sakr, Charbel, Yongjune Kim, and Naresh Shanbhag. "Analytical Guarantees on Numerical Precision of Deep Neural Networks." International Conference on Machine Learning. 2017.

@inproceedings{sakr2017analytical,

title={Analytical Guarantees on Numerical Precision of Deep Neural Networks},

author={Sakr, Charbel and Kim, Yongjune and Shanbhag, Naresh},

booktitle={International Conference on Machine Learning},

pages={3007--3016},

year={2017}

}

Charbel 
