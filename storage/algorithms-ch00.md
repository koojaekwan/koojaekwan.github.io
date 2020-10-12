# 2 Linear Algebra

What is Algebra?

- To formalize intuitive concepts, we construct a set of objects (symbols) and a set of rules to manipulate these objects.

Especially, What is Linear Algebra?

- object → vectors

What is Vector in general?

- any object that satisfies two properties (addtion, scalar multiplication)

- ex) Geometric vectors, Ponlynomials, Audio Signals, Elements of R^n

![image-20200505200249294](/Users/skcc10170/Library/Application Support/typora-user-images/image-20200505200249294.png)

<br/>

## 2.1 Systems of Linear Equations

System of linear equations

solution

- has 3 types
- no solution, unique solution, infinitely many solutions
- Geometric Interpretation of Systems of Linear Equations

(bridge) For a systematic appraoch to solving systems of LE, we introduce a useful compact notation, matrices

<br/>

## 2.2 Matrices

matrix, row (vector), column (vector)

### 2.2.1 Matrix Addition and Multiplication

A+B, AB

- neighboring dimension match, AB=C (nk, km => nm)
- dot product between two vectors (matrix 곱셈식에서...찾아볼 수 있음)
- not commutative
- remark) Hadamard product

Identity Matrix

matrices property

- associativty
- distributivity
- multiplication with the identity matrix

<br/>

### 2.2.2 Inverse and Transpose

Inverse

- regular/invertible/nonsingular

- singular/noninvertible

- exists then unique

transpose

symmetric

### 2.2.3 Multiplication by a Scalar

associativity

distribuitivity

### 2.2.4 Compact Representation of Systems of Linear Equations

a system of LE, compactly represented in matrix from Ax=b

<br/>

## 2.3 Solving Systems of Linear Equations

### 2.3.1 Particular and General Solution

particular solution, special solution

general solution

3 steps ; general approach for finding the solution for LE

(bridge) need constructive algorithmic way of transforming any system of Les into particularly simple form : GE

- elementary transformations
- after GE, then apply three steps form

### 2.3.2 Elementary Transformations

elementary transformations

- exchange of two equations
- multiplication of an equation with a constant
- addition of two equtions

augmented matrix

REF

- pivot
  - basic variables
  - free variables
  - 





### 2.3.3 The Minus-1 Trick

<br/>

## 2.4 Vector Spaces

2.4.1 Groups

2.4.2 Vector Spaces

2.4.3 Vector Subspaces

<br/>

## 2.5Linear Indepedence

<br/>

## 2.6 Basis and Rank

2.6.1 Generating Set and Basis

2.6.2 Rank

<br/>

## 2.7 Linear Mappings

2.7.1 Matrix Representation of Linear Mappings

2.7.2 Basis Change

2.7.3 Image and Kernel

<br/>

## 2.8 Affine Spaces

2.8.1 Affine Subspaces

2.8.2 Affine Mappings

<br/>

<br/>

<br/>

<br/>

<br/>



# 4 Matrix Decompositions

## 4.1 Determinant and Trace

## 4.2 Eigenvalues and Eigenvectors

## 4.3 Cholesky Decomposition

## 4.4 Eigendecomposition and Diagonalization



## 4.5 Singular Value Decomposition

### 4.5.1 Geometric Intuitions for the SVD

### 4.5.2 Construction of the SVD



## 4.6 Matrix Approximation



## 4.7 Matrix Phylogeny



## 4.8 Further Reading



![image-20200503011229862](/Users/skcc10170/Library/Application Support/typora-user-images/image-20200503011229862.png)

[https://github.com/bikestra/bikestra.github.com/blob/master/notebooks/Convex%20Conjugates.ipynb?fbclid=IwAR2143vFu2bDYVBFSHQ-i_-YrY6NHfCaZ81o21q1ZgFKO9pj2ExU_P1EfzU](https://github.com/bikestra/bikestra.github.com/blob/master/notebooks/Convex Conjugates.ipynb?fbclid=IwAR2143vFu2bDYVBFSHQ-i_-YrY6NHfCaZ81o21q1ZgFKO9pj2ExU_P1EfzU)





https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-253-convex-analysis-and-optimization-spring-2012/lecture-notes/MIT6_253S12_lec_comp.pdf



https://people.eecs.berkeley.edu/~wainwrig/stat241b/lec10.pdf

http://web.stanford.edu/class/cs224n/readings/cs229-cvxopt.pdf