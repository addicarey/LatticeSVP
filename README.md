# LatticeSVP
Advancing Post-Quantum Lattice-based Cryptography by developing Efficient Shortest Vector Problem Approaches.

Purpose 

The purpose of this project was to create and evaluate a heuristic algorithm for the Shortest Vector Problem (SVP), a fundamental challenge in lattice-based cryptography. The engineering goal was to integrate an algorithm with a refined selection method that could find short vectors in lattices more effectively than basic methods. It was expected that this algorithm would give results similar to traditional brute-force search in small lattices, but with potential to scale better as dimensions increase.

Method 

The algorithm was written in Python and tested on randomly generated lattices between three and six dimensions. Before execution, the lattice bases were conditioned to improve their structure. The refined algorithm was then compared with a brute-force solver by examining the length of the vector it produces, its runtime, and how well it handled complex lattice conditions. Randomness was assessed for using multiple trials.

Results 

The refined algorithm reliably found short vectors of similar quality to brute-force, and in some challenging cases, it out-performed brute-force algorithms. Although this method experienced higher runtime in lower dimensions, it displayed statistically significant improvements in reliability when the lattices were harder to solve. Results varied between executions due to its randomised design, but overall patterns remained consistent.

Conclusion 

The project successfully met its goal of developing a new heuristic solver for SVP. The results suggest this could become a useful approach for higher-dimensional lattices, where traditional methods are infeasible. This work provides a foundation for future research in post-quantum cryptography, where efficient solvers are critical.

