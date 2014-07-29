SVD-UPDATE: Fast recalculation of SVD for new attached column (row)
=====================================================================

This module implements SVD update i. e. fast recalculation of SVD for
new matrix column (row).

Important references for this method are given in:

1. [1] Gu, M. & Eisenstat, S. C. "A Stable and Fast Algorithm for Updating the
 Singular Value Decomposition", Yale University, 1993
 
2. [2] Brand, M. "Fast low-rank modifications of the thin singular value
decomposition", Linear Algebra and its Applications , 2006, 415, 20 - 30

3. [3] Stange, P. "On the Efficient Update of the Singular Value Decomposition
Subject to Rank-One Modifications", 2009


Install:
------------------------------------------

For this package to work only Numpy, Scipy and Matplotlib are required. Matplotlib is used
only in testing functions. However, Scipy need to be compiled from sources in order to
use some LAPACK function "dlasd4" which are not exposed originally.

To include this function follow these steps:
1. Copy the file "For_scipy_modification_flapack.pyf.src" from this package
to {path to scipy source directory}/scipy/linalg/. 
2. Rename the present file "flapack.pyf.src" (for instance to "flapack.pyf.src.old") to restore it if needed.

   Actually this, file has been used for Scipy version 0.12.0c1. If you use older verions of scipy you can
   track modifications (there are not many) in file "For_scipy_modification_flapack.pyf.src" and add 
   those to the "flapack.pyf.src" in the new distibution. 
 
3. And rename the copied file to "flapack.pyf.src"
4. Build and install scipy using standard instructions.

Usage:
------------------------------------------

There are two interfaces to SVD update.

* The function "SVD_update" is used to update SVD only once. It returns SVD of a matrix with one extra column (or
  row) attached.

* The class SVD_updater is used to perform many sequential updates. It uses some extra techniques to speed up
  computations in sequential updates. These techniques are presented in the paper [2]. 

