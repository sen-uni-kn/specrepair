# DeepOpt: Scalable Specification-based Falsification of Neural Networks using Black-Box Optimization

DeepOpt: Scalable Specification-based Falsification of Neural Networks using Black-Box Optimization

Authors: Fabian Bauer-Marquart, Stefan Leue, Christian Schilling

29th International Joint Conference on Artificial Intelligence (IJCAI'20)

## Abstract

Decisions made by deep neural networks (DNNs) have a tremendous impact on the dependability of the systems that they are embedded into, which is of particular concern in the realm of safety-critical systems.
Property-verification techniques for DNNs based on SMT (satisfiability modulo theories) or abstraction have been shown to be computationally expensive.
The key idea is to algebraically combine the network and a formal specification of its input and output constraints.
The proposed falsification method is based on a black-box optimization method called simplicial homology global optimization (SHGO) [Endres et al., 2018].
Counterexamples are detected using a refinement approach that adapts to the DNN’s output value distribution.
The approach does not require original training data and is applicable to more types of activation functions than many state-of-the-art ap- proaches.
Our proposed method has been implemented and evaluated on networks of varying sizes.
Experiments demonstrate superior scalability compared to other approaches since it is indifferent to the hidden neuron count.
At the same time, our method shows a precision comparable to that of existing alternative methods, making it well-suited for fast counterexample generation during system design.


## Results
