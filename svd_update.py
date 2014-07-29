# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:40:48 2013

@author: Alexander Grigorievskiy

Important references are:

[1] Gu, M. & Eisenstat, S. C. "A Stable and Fast Algorithm for Updating the
 Singular Value Decomposition", Yale University, 1993
 
[2] Brand, M. "Fast low-rank modifications of the thin singular value
decomposition", Linear Algebra and its Applications , 2006, 415, 20 - 30

[3] Stange, P. "On the Efficient Update of the Singular Value Decomposition
Subject to Rank-One Modifications", 2009 
"""

import numpy as np
import scipy as sp
import scipy.optimize as opt
import numpy.random as rnd
import time
import time
import scipy.io as io

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

rounding_ndigits = 14 # number of rounding digits of sigmas. Need to round sigmas
                      # in order to compare properly the float numbers. 
                      # Used also for zeros detection.
                      # Machine epsilon for float number is 2.22e-16

epsilon1 = 10**(-rounding_ndigits) # epsilon for detecting zeros

class Timer(object):
    """
    Timer context manager. It is used to measure running time during testing.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs

def extend_matrix_by_one(M):
    """
    This function extendes the matrix M by adding new row and new column to the end.
    Column and row consist of zeros except the one on the diagonal.
    """

    M = np.hstack( (M, np.zeros((M.shape[0],1 ) ) ) )
    M = np.vstack( (M, np.array( (0.0,)*(M.shape[1]-1) + (1.0,), ndmin=2 ) ) )

    return M
    
class SVD_updater(object):
    """
    This class is intended to perform sequential SVD update. So, it should
    be used if we want to sequentially update SVD attaching many columns sequentially.
    
    If we need to update SVD (by extra column) only once function update_SVD should be used.
    """

    def __init__(self, U,S,Vh, update_V = False, reorth_step=100):
        """
        Class constructor.

        Input:
            U,S,Vh - current SVD
            update_V - whether or not update matrix V
            reorth_step - how often to perform orthogonalization step
            
        Output:
            None
        """
        
        self.outer_U = U
        self.outer_Vh = Vh
        
        self.inner_U = None        
        self.inner_Vh = None        
        self.inner_V_new_pseudo_inverse = None # pseudo inverse of the inner_V        
        
        self.S = S
        self.m = U.shape[0] # number of rows. This quantity doe not change
        self.n = Vh.shape[1] # number of columns. This quantity changes every time we add column

        self.update_Vh = update_V
        #assert update_V == False, "Updarting V is corrently not supported"
        
        self.reorth_step = reorth_step
        self.update_count = 0 # how many update columns have been called. 
        self.reorth_count = 0 # how many update columns since last reorthogonalizetion have been called
                              #This counter is needed to decide when perform an orthogonalization
        
    def add_column(self, new_col):
        """
        Add column to the current SVD.
        
        Input:
            new_col - one or two dimensional vector ( if two dimensional one it must have the shape (len,1) )
            
        Output:
            None
        """        
        
        #self.zero_epsilon = np.sqrt( np.finfo(np.float64).eps * self.m**2 * self.n * np.max( new_col)*2) # epsilon to determine when rank not increases
        self.zero_epsilon = np.sqrt( np.finfo(np.float64).eps * self.m**2 * self.n * 10*2) # epsilon to determine when rank not increases
        
        old_shape = new_col.shape        
        new_col.shape = (old_shape[0],)
        
        m_vec = np.dot(self.outer_U.T, new_col) # m vector in the motes    
        if not self.inner_U is None:
            m_vec = np.dot( self.inner_U.T, m_vec)
            
        if self.n < self.m: # new column
            if not self.inner_U is None:
                new_u = new_col -  np.dot( self.outer_U, np.dot(  self.inner_U, m_vec ) )
            else:                    
                new_u = new_col - np.dot( self.outer_U, m_vec )
                
            mu = sp.linalg.norm( new_u, ord=2 )
        
            # rank is not increased. 
            if (np.abs(mu) < self.zero_epsilon ): # new column is from the same subspace as the old column
                rank_increases = False
                U1, self.S, V1 = _SVD_upd_diag( self.S, m_vec, new_col=False )
            
            else: # rank is increased
                if np.abs(mu) < 0.1: # temporary check
                    pass
                
                rank_increases = True
                S = np.concatenate( (self.S, (0.0,) ) )       
                m_vec = np.concatenate((m_vec, (mu,) ))    
            
                U1, self.S, V1 = _SVD_upd_diag( S, m_vec, new_col=True)
                
                # Update outer matrix
                self.outer_U = np.hstack( (self.outer_U, new_u[:,np.newaxis] / mu) ) # update outer matrix in case of rank increase             
        
            if self.inner_U is None:
                self.inner_U = U1
            else: 
                if rank_increases:
                    self.inner_U = extend_matrix_by_one(self.inner_U)
                    
                self.inner_U = np.dot( self.inner_U, U1)
            
            if self.update_Vh:
                if rank_increases:
                    self.outer_Vh = extend_matrix_by_one(self.outer_Vh)
                    if not self.inner_Vh is None:
                        self.inner_Vh = np.dot( V1.T, extend_matrix_by_one(self.inner_Vh) )
                    else:
                        self.inner_Vh = V1.T
                        
                    if not self.inner_V_new_pseudo_inverse is None:
                        self.inner_V_new_pseudo_inverse = np.dot( V1.T, extend_matrix_by_one(self.inner_V_new_pseudo_inverse) ) 
                    else:
                        self.inner_V_new_pseudo_inverse = V1.T
                else:
                    r = V1.shape[1]
                    W = V1[0:r,:]
                    w = V1[r,:]; w.shape= (1, w.shape[0]) # row
                    
                    w_norm2 = sp.linalg.norm(w, ord=2)**2
                    W_pseudo_inverse = W.T + np.dot( (1/(1-w_norm2))*w.T, np.dot(w,W.T) )
                    del r, w_norm2
                    
                    if not self.inner_Vh is None:
                        self.inner_Vh = np.dot( W.T, self.inner_Vh)
                    else:
                        self.inner_Vh = W.T
                    
                    if not self.inner_V_new_pseudo_inverse is None:
                        self.inner_V_new_pseudo_inverse = np.dot(W_pseudo_inverse, self.inner_V_new_pseudo_inverse)
                    else:
                        self.inner_V_new_pseudo_inverse = W_pseudo_inverse
                    
                    self.outer_Vh = np.hstack( (self.outer_Vh, np.dot( self.inner_V_new_pseudo_inverse.T, w.T)) )
                        
                    del w, W_pseudo_inverse, W
                    
#            U = np.dot(U, U1)
#            Vh = np.dot(V1.T,Vh) # V matrix. Need to return V.T though
                    
        else: #  self.n > self.m            
                            
            U1, self.S, V1 = _SVD_upd_diag( self.S, m_vec, new_col=False)
                            
            if self.inner_U is None:
                self.inner_U = U1
            else: 
                self.inner_U = np.dot( self.inner_U, U1)        
                    
            if self.update_Vh:
                r = V1.shape[1]
                W = V1[0:r,:]
                w = V1[r,:]; w.shape= (1, w.shape[0]) # row
                
                w_norm2 = sp.linalg.norm(w, ord=2)**2
                W_pseudo_inverse = W.T + np.dot( (1/(1-w_norm2))*w.T, np.dot(w,W.T) )
                del r, w_norm2
                
                if not self.inner_Vh is None:
                    self.inner_Vh = np.dot( W.T, self.inner_Vh )
                else:
                    self.inner_Vh = W.T
                
                if not self.inner_V_new_pseudo_inverse is None:
                    self.inner_V_new_pseudo_inverse = np.dot(W_pseudo_inverse, self.inner_V_new_pseudo_inverse)
                else:
                    self.inner_V_new_pseudo_inverse = W_pseudo_inverse
                
                self.outer_Vh = np.hstack( (self.outer_Vh, np.dot( self.inner_V_new_pseudo_inverse.T, w.T) ) )
                    
                del w, W_pseudo_inverse, W
          
        self.n = self.outer_Vh.shape[1]
        new_col.shape  = old_shape
        
        self.update_count += 1 # update update step counter
        self.reorth_count += 1
        
        if self.reorth_count >= self.reorth_step:
            self._reorthogonalize()
            self.reorth_count = 0
        #return np.dot( self.outer_U, self.inner_U), self.S, np.dot( self.inner_Vh, self.outer_Vh)


    def get_current_svd(self):
        """
        Function computes the currrent SVD, returns the result
        and updates outer matrices by multipying by inner matrices.
        
        Output:
            U, S, Vh - SVD. Vh is none if it is not updated.
        """        
        
        if not self.inner_U is None:
            Ur = np.dot( self.outer_U, self.inner_U)
        else:
            Ur = self.outer_U
            
        if self.update_Vh:
            if not self.inner_Vh is None:
                Vhr = np.dot( self.inner_Vh, self.outer_Vh)
            else:
                Vhr = self.outer_Vh
        else:
            Vhr = None
        
       
        return Ur, self.S, Vhr
     
        
    def _reorthogonalize(self):
        """
        Uses orthogonalization method mentioned in:            
        Brand, M. "Fast low-rank modifications of the thin singular value
        decomposition Linear Algebra and its Applications" , 2006, 415, 20 - 30.
        
        Actually the usefulness of this method is not wel justified. But it is 
        implemeted here. This function is called from "add_column" method.
        """

        if not self.inner_U is None:
            US = np.dot( self.inner_U, np.diag(self.S) )
            (Utmp,Stmp,Vhtmp) = sp.linalg.svd( US  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
            
            self.inner_U = Utmp
            self.S = Stmp
            
            # update outer matrix ->            
            self.outer_U = np.dot( self.outer_U, self.inner_U)
            self.inner_U = None
            # update outer matrix <-
            
            if self.update_Vh:
                self.inner_Vh = np.dot( Vhtmp, self.inner_Vh)
                
                # update outer matrix ->            
                self.outer_Vh = np.dot( self.inner_Vh, self.outer_Vh)
                self.inner_Vh = None
                self.inner_V_new_pseudo_inverse = None
                # update outer matrix <-
    
            if self.update_Vh:
                return self.outer_U, self.S, self.outer_Vh
            else:
                return self.outer_U, self.S, None
        else:
            pass
    
    def reorth_was_pereviously(self):
        """
        Function for external testing.
        
        Functions returns the boolean value which tells 
        whether reorthogonalization has performed on the previous step.
        """
    
        return (self.reorth_count == 0)    
    
    
    
def test_SVD_updater():
    """
    Testinf function for SVD_updater class.
    """
    
#    # Test column update. Thin matrix, rank increases 
#    A = rnd.rand(5,3)
#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#        
#    a1 = rnd.rand(5,1)
#    A = np.hstack( (A,a1) )
#    (Ut1,St1,Vht1) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#
#    SVD_upd = SVD_updater( U,S,Vh, update_U=True, update_V = True, reorth_step=10)
#
#    (Us1, Ss1, Vhs1) = SVD_upd.add_column(a1 )
#    Ar1 = np.dot( Us1, np.dot(np.diag(Ss1), Vhs1 ) )
#
#    diff1 = np.max( np.abs( A - Ar1) )/St1[0]
#    
#    a2 = rnd.rand(5,1)
#    A = np.hstack( (A,a2) )
#    
#    (Us2, Ss2, Vhs2) = SVD_upd.add_column( a2 )
#    
#    Ar2 = np.dot( Us2, np.dot(np.diag(Ss2), Vhs2 ) )
#
#    diff2 = np.max( np.abs( A - Ar2) )/St1[0]
    
        
    # Test column update. Thin matrix, rank not increases 
#    A = rnd.rand(5,3)
#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#        
#    a1 = rnd.rand(5,1)
#    #a1 = np.dot(A, np.array([2,1,4],ndmin=2 ).T ) 
#    A = np.hstack( (A,a1) )
#    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#
#    SVD_upd = SVD_updater( U,S,Vh, update_U=True, update_V = True, reorth_step=10)
#
#    (Us1, Ss1, Vhs1) = SVD_upd.add_column(a1 )
#    Ar1 = np.dot( Us1, np.dot(np.diag(Ss1), Vhs1 ) )
#
#    diff1 = np.max( np.abs( A - Ar1) )/St[0]
#    
#    a2 = rnd.rand(5,1)
#    #a2 = np.dot(A, np.array([2,1,4,-3],ndmin=2 ).T ) 
#    A = np.hstack( (A,a2) )
#    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#    (Us2, Ss2, Vhs2) = SVD_upd.add_column( a2 )
#    
#    Ar2 = np.dot( Us2, np.dot(np.diag(Ss2), Vhs2 ) )
#
#    diff2 = np.max( np.abs( A - Ar2) )/St[0]
#    
#    return diff2

    # Test column update. Fat matrix 
    #A = rnd.rand(5,4)
#    A = rnd.rand(5,5)
#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#        
#    a1 = rnd.rand(5,1)
#    #a1 = np.dot(A, np.array([2,1,4],ndmin=2 ).T ) 
#    A = np.hstack( (A,a1) )
#    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#
#    SVD_upd = SVD_updater( U,S,Vh, update_U=True, update_V = True, reorth_step=10)
#
#    (Us1, Ss1, Vhs1) = SVD_upd.add_column(a1 )
#    Ar1 = np.dot( Us1, np.dot(np.diag(Ss1), Vhs1 ) )
#
#    diff1 = np.max( np.abs( A - Ar1) )/St[0]
#    
#    a2 = rnd.rand(5,1)
#    #a2 = np.dot(A, np.array([2,1,4,-3],ndmin=2 ).T ) 
#    A = np.hstack( (A,a2) )
#    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#    (Us2, Ss2, Vhs2) = SVD_upd.add_column( a2 )
#    
#    Ar2 = np.dot( Us2, np.dot(np.diag(Ss2), Vhs2 ) )
#
#    diff2 = np.max( np.abs( A - Ar2) )/St[0]
#    
#    return diff2

    # test function SVD update

    A = rnd.rand(5,3)
    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True,
                             overwrite_a=False, check_finite=False )
        
    #a1 = rnd.rand(5,1)
    a1 = np.dot(A, np.array([2,1,4],ndmin=2 ).T ) 
    A = np.hstack( (A,a1) )
    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True,
                                      overwrite_a=False, check_finite=False )

    (Us1, Ss1, Vhs1) = update_SVD( U, S, Vh, a1, a_col_col=True)
    Ar1 = np.dot( Us1, np.dot(np.diag(Ss1), Vhs1 ) )

    diff1 = np.max( np.abs( A - Ar1) )/St[0]

    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True,
                                  overwrite_a=False, check_finite=False )
    
    #a2 = np.array([2,1,4,7],ndmin=2 )
    a2 = np.array([0,0,0,0],ndmin=2 )
    A = np.vstack( (A,a2) )
    
    (Us2, Ss2, Vhs2) = update_SVD( U, S, Vh, a2, a_col_col=False)
    Ar2 = np.dot( Us2, np.dot(np.diag(Ss2), Vhs2 ) )

    diff2 = np.max( np.abs( A - Ar2) )/St[0]

    return diff1

def test_SVD_update_reorth(n_rows,start_n_col, n_max_cols, prob_same_subspace,
                           reorth_step):
    """
    Test how the orthogonality property changes of updated SVD.    
    
    Inputs:
        n_rows - how many rows in the initial matrix (this value is constant during iterations)
        start_n_col - how many columns in the initial matrix (columns are added sequentially)
        n_max_cols - numbers of columns to add.
        prob_same_subspace - probability that the column is from the current column subspace
        reorth_step - how often to do reorthogonalization.
    """
    import numpy.random as rnd
    #n_rows = 1000   
    #n_max_cols = 1000
    
    
#    (X,Y) = ds.make_regression(n_samples = 1000, n_features = 7,n_informative=5, \
#                       n_targets=2, bias = 2.0, effective_rank = 2)                            
    A = rnd.rand(n_rows, start_n_col)
    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
    
    svd_upd = SVD_updater( U,S,Vh, update_V = True, reorth_step=reorth_step)
   
    svd_comp = np.empty((n_max_cols, 9))
    times = np.empty((n_max_cols, 4))    
    
    same_sub = False
    for ii in xrange( 0,n_max_cols ):
        if prob_same_subspace >= rnd.rand(): # column from the same subspace
            a1 = rnd.rand(A.shape[1],1)
            a1 = np.dot(A, a1)
            a1 = a1/ np.max(a1)
            same_sub = True
        else:
            a1 = rnd.rand(n_rows,1)
            same_sub = False
        A = np.hstack( (A,a1) )
        
        with Timer() as t:
            svd_upd.add_column(a1)
        times[ii,0] = t.msecs
        
        with Timer() as t:
            (Ut, St, Vht) = svd_upd.get_current_svd()
            
        times[ii,1] = t.msecs
        times[ii,2] = times[ii,0] + times[ii,1]
        
        with Timer() as t:
            (Us,Ss,Vhs) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
        times[ii,3] = t.msecs
        
        tmp = np.abs( Ss[0:St.shape[0]] - St ) / Ss[ 0:St.shape[0] ]
        t1 = np.max( tmp )
        t_pos = np.nonzero( tmp == t1 )[0][0]
        svd_comp[ii,0] = t1
        svd_comp[ii,1] = t_pos
        svd_comp[ii,2] = np.sqrt( A.shape[0]* A.shape[1] )
        svd_comp[ii,3] = A.shape[0]
        svd_comp[ii,4] = A.shape[1]
        svd_comp[ii,5] = np.int( svd_upd.reorth_was_pereviously() )
        del tmp, t_pos
        
        Ar1 = np.dot( Ut, np.dot(np.diag(St), Vht ) )
        svd_comp[ii,6] =  np.max( np.abs( Ar1 - A ) )
        svd_comp[ii,7] =  np.max( np.abs(np.dot( Ut.T, Ut ) - np.eye( Ut.shape[1]) ) )    
        svd_comp[ii,8] = np.max( np.abs(np.dot( Vht, Vht.T ) - np.eye( Vht.shape[0]) ) )
    
        print ii
        
#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#    Ar2 = np.dot( U, np.dot(np.diag(S), Vh ) )
    
    t1 = svd_comp; t2 = times
    
    result = {}
    result['svd_comp'] = svd_comp
    result['times'] = times
    result['n_rows'] = n_rows
   
    io.savemat('inc_svd_update_6.mat', result )
    
    # Plot
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)         
    plt.figure(1)
    plt.plot( t1[:,4], t2[:,2]/1000.0, 'rs-', label='SVD update')
    plt.plot( t1[:,4], t2[:,3]/1000.0, 'bo-', label='New SVD')
    plt.plot( t1[:,4], t2[:,0]/1000.0, 'm--', 
              label='SVD update: except final matrix multiplication')
    plt.plot( t1[:,4], t2[:,1]/1000.0, 'y-.',
              label='SVD update: only final matrix multiplication')
    plt.legend(loc=2,prop={'size':16})    
    plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
    plt.suptitle('Comp. time of sequential SVD update, its components and new SVD', fontsize=20)
    plt.xlabel( 'Number of columns' , fontsize = 17 )
    plt.ylabel( 'Seconds' , fontsize = 17 )    
    plt.show()
    #plt.savefig('Seq_SVD_update_2.png')
    
    plt.close()
    
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)  
    plt.figure(2)
    plt.plot( t1[:,4], t1[:,0], 'bo-')
    plt.suptitle('Squential SVD. Maximum relative singular value differences',
                 fontsize=20)
    plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
    plt.ylabel('Difference', fontsize = 18 )
    plt.xlabel('Number of columns', fontsize = 18 )    
    #plt.savefig('Seq_SVD_sing_val_diff_2.png')    
    plt.show()
    #plt.close()   
    
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)  
    plt.figure(3)
    plt.plot(t1[:,4], t1[:,4], 'bo-', 
              label='Total number of singular values')
    plt.plot(t1[:,4], t1[:,1], 'rs-',
              label='Index of maximum difference singular value')
    plt.legend(loc=2, prop={'size':16})
    plt.suptitle('Squential SVD. Index of maximum difference singular value',
                  fontsize = 20 )
    plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17)
    plt.ylabel('Index', fontsize = 18 )
    plt.xlabel('Number of columns', fontsize = 18)   
    plt.show()
    
    return (svd_comp,times)
    
    xx = [2784, 2999, 4752, 6344]
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)  
    plt.figure(3)
    plt.plot( xx, [14.69, 15.83, 23.94, 32.87], 'bo-', linewidth=3, ms=10,
              label='OP-ELM')
    plt.plot( xx, [12.67, 10.09, 14.77, 14.53], 'rs-', linewidth=3, ms=10,
              label='Inc (OP)-ELM')
    plt.legend(loc=2,prop={'size':18})
    plt.suptitle('Running Time with Respect to Number of Samples',
                  fontsize = 20)
    #plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
    plt.ylabel('Seconds', fontsize = 18)
    plt.xlabel('Number of Samples', fontsize = 18)   
    plt.show()
    
    xx = [100,300,600]
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)  
    plt.figure(3)
    plt.plot(xx, [1.64, 15.83, 65.41], 'bo-', linewidth=3, ms=10,
             label='OP-ELM')
    plt.plot(xx, [1.33, 10.09, 38.96], 'rs-', linewidth=3, ms=10,
             label='Inc (OP)-ELM')
    plt.legend(loc=2,prop={'size':18})
    plt.suptitle('Running Time with Respect to Initial Number of Neurons',
                 fontsize = 20 )
    #plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
    plt.ylabel('Seconds', fontsize = 18 )
    plt.xlabel('Initial Number of Neurons', fontsize = 18 )   
    plt.show()
    
    
func_calls = []
it_count = []
#@profile    
def one_root(func, a_i, b_i, ii,sigmas,m_vec):
    """
    Function is used to find one root on the interval [a_i,b_i] a_i < b_i. It must be
    so that f(a_i) < 0, f(b_i) > 0. The reason for this function is that for new singular
    values values at a_i and b_i f is infinity. So that procedure is needed to find correct
    intervals.
    
    Derivative free method brentq is used internally to find root. Maybe some
    other method would work faster.
    
    Inputs:
        func - function which can be called
        a_i - left interval value
        b_i - right interval value
        
    Output:
        Root on this interval
    """
    
    shift = 0.01    
    delta = b_i - a_i 
    tmp =  shift*delta       
        
    eps = np.finfo(float).eps
        
    a_new = a_i + tmp
    b_new = b_i - tmp
    
    # a_i * eps  - absolute error
    a_max_it = np.ceil( np.log10( tmp / (a_i * eps) ) ) + 2
    b_max_it = np.ceil( np.log10( tmp / (b_i * eps) ) ) + 2   
    
    if np.isnan( a_max_it ) or a_max_it > 20:
        a_max_it = 17
    
    if np.isnan( b_max_it ) or b_max_it > 20:
        b_max_it = 17
        
    a_found = False; b_found = False; it = 0
    while not ( a_found and b_found ):
        shift /= 10
        if not a_found:
            if func(a_new) >= 0:
                a_new = a_i + shift*delta
                
                if it >= a_max_it:
                    return a_new
            else:
                a_found = True
                
        if not b_found:
            if func(b_new) <= 0:
                b_new = b_i - shift*delta
                
                if it >= b_max_it:
                    return b_new
            else:
                b_found = True
        it += 1 
 
    res = opt.brentq( func, a_new, b_new, full_output=True, disp=False)
    if res[1].converged == False:
        pass
    
    func_calls.append( res[1].function_calls )
    it_count.append( it)
    return res[0]
#    x0 = 0.5*(b_i - a_i)    
#    
#    return opt.newton(func, x0, fprime = func_der1, fprime2 = func_der2)

#@profile
def find_roots(sigmas, m_vec, method=1):
    """
    Find roots of secular equation of augmented singular values matrix
    
    Inputs:
        sigmas - (n*n) matrix of singular values, which are on the diagonal.                
                
        m_vec - additional column vector which is attached to the right of the Sigma.
                Must be (m*1)
        method - which method to use to find roots
    
    There are two ways to find roots for secular equation of augmented s.v.
    matrix. One way is to use again SVD decomposition and the second method
    is to find roots of algebraic function on a certain interval.
    Currently method 2 is used by using imported lapack function.
    """

    #sigmas = np.diag(Sigma) # now singular values are from largest to smallest
    
    if method == 1: # using interlacing properties find zeros on the intervals
        m_vec2 = np.power(m_vec,2)
        sig2 = np.power(sigmas,2)       
        #func = lambda l: 1 + np.sum( m_vec2 / ( ( sigmas + l) * (sigmas - l) ) )
        func = lambda l: 1 + np.sum( m_vec2 / ( sig2 - l**2 ) )
        #func_der1 = lambda l: 2l*sum( m_vec2 / np.power(( sigmas + l) * (sigmas - l),2 ) )
        #func_der2 = lambda l: 2*sum( m_vec2 / np.power(( sigmas + l) * (sigmas - l),2 ) ) + 8*l*l*sum( m_vec2 / np.power(( sigmas + l) * (sigmas - l),3 ) )     
        
        if (sigmas[-1] < epsilon1) and (sigmas[-2] < epsilon1): # two zeros at the end        
            it_len = len(sigmas) - 1
            append_zero = True
        else:
            it_len = len(sigmas)
            append_zero = False            
            
        roots = [] # roots in increasing order (singular values of new matrix)
        # It is assumed that the first eigenvalue of Sigma [(n+1)*(n+1)] is zero
        for i in xrange(0, it_len ):
            
            if (i == 0):            
                root = one_root(func, sigmas[0], (sigmas[0] + np.sqrt(np.sum(m_vec2)) ),i,sigmas,m_vec )
            else:
                root = one_root(func, sigmas[i], sigmas[i-1],i,sigmas,m_vec )
                
            roots.append( root )
        if append_zero:
            roots.append( 0.0 )
            
        return np.array(roots)
        
    if method == 2: # Imported lapack function
        it_len = len(sigmas)
        sgm = np.concatenate( ( sigmas[::-1], (sigmas[0] + it_len*np.sqrt(np.sum( np.power(m_vec,2) )) ,) ) )
        mvc = np.concatenate( ( m_vec[::-1],    (0,) ) )
        roots = []
        
        if (sigmas[-1] < epsilon1) and (sigmas[-2] < epsilon1): # two zeros at the end        
            it_start = 2
            prepend_zero = True
        else:
            it_start = 1
            prepend_zero = False          
        
        #sigmas_minus_i = [] # ( sigmas - new_sigmas[i] ) needed for singular vector construction
        #sigmas_plus_i = []  # ( sigmas + new_sigmas[i] ) needed for singular vector construction
        for i in xrange(it_start, it_len ): # find all singular except the last
            res = sp.linalg.lapack.dlasd4(i, sgm, mvc )
         #   sigmas_minus_i.append( res[0][0:-1] )
            roots.append( res[1] )
         #   sigmas_plus_i.append( res[2][0:-1] )
        
        # find last singular value ->
        max_iter_last = 10; iter_no = 0
        exit_crit = False
        while not exit_crit:
            res = sp.linalg.lapack.dlasd4( it_len, sgm, mvc )
            
            if (res[3] == 0) or (iter_no >= max_iter_last):
                exit_crit = True
            else:
                sgm[-1] = 10 * sgm[-1]            
                iter_no += 1            
            
        if  (res[3] > 0) or np.any( np.isnan(roots) ):
            pass            
        else:
            if iter_no > 1:
                print "Iters:  ", iter_no
            roots.append( res[1] ) # append the last singular value
        # find last singular value <-
            
        if prepend_zero:
            roots = [0.0,] + roots
            
        return np.array(roots[::-1])
        
    if method == 3: # by eigh call 
        M = np.diag( np.power( sigmas, 2) )  + np.dot( m_vec[:, np.newaxis],  m_vec[np.newaxis, : ] )        
        #sm = sp.linalg.svd(M, full_matrices=False, compute_uv=False, overwrite_a=True, check_finite=False)
        
        sm = sp.linalg.eigh(M, eigvals_only=True, overwrite_a=True, check_finite=False)
        sm = sm[::-1]
        del M
        
        return np.sqrt( sm )
        
    if method == 4: # by svd call 
        M = np.diag( np.power( sigmas, 2) )  + np.dot( m_vec[:, np.newaxis],  m_vec[np.newaxis, : ] )        
        sm = sp.linalg.svd(M, full_matrices=False, compute_uv=False, overwrite_a=True, check_finite=False)
        
        #sm = sp.linalg.eigh(M, eigvals_only=True, overwrite_a=True, check_finite=False)
        del M
 
        return np.sqrt( sm )
    
def _SVD_upd_diag_equal_sigmas( sigmas, m_vec , new_col):
    """    
    This is internal function which is call by _SVD_upd_diag. It analyses
    if there are equal sigmas in the set of sigmas and if there are returns
    the appropriate unitary transformation which separates unique and equal
    sigmas. This is needed because the method in _SVD_upd_diag works only for
    unique sigmas. It also detects zeros in the m_vec and performs appropriate
    permutation if these zeros are found. When new column is added the 
    original matrix is square and it is mandatory that sigma[-1] = 0 and
    m_vec[-1] != 0. When new row is added original matrix is not square
    and sigmas are arbitary.
    
    Inputs:
        sigmas - singular values of a square matrix
        m_vec - extra column added to the square diagonal matrix of singular
                vectors.
        new_col - says which task is task is solved. True if originally the 
                  problem of new column is solved, False if new row is added.
    Output:
        is_equal_sigmas - boolean which is True if there are equal sigmas,
                          False otherwise.
        uniq_length - quantity of unique sigmas.
        U - unitary transformation which transform the 
            original matrix (S + mm*) into the form 
            where equal sigmas are at the end of the diagonal of the new matrix.
    """
    
    orig_sigma_len = len(sigmas) # length ofsigma and m_cel vectors
    U = None # unitary matrix which turns m_vec into right view in case repeated sigmas and/or zeros in m_vec
    z_col = m_vec.copy()
    
    if new_col: # temporary change the last element in order to avoid
                # processing if there are other zero singular values
        sigmas[-1] = -1
    indexed_sigmas = [s for s in enumerate( sigmas ) ] # list of indexed sigmas    
    
    
    # detect zero components in m_vec ->        
    zero_inds = [] # indices of elements where m_col[i] = 0
    nonzero_inds = [] # indices of elements where m_col[i] != 0
    for (i,m) in enumerate( np.abs(m_vec) ):
        if (m < epsilon1): # it is considered as zero
            zero_inds.append(i)
        else:
            nonzero_inds.append(i)
            
    num_nonzero = len( nonzero_inds )
    
    U_active = False
    if zero_inds:
        U_active = True
    
    indexed_sigmas =  [indexed_sigmas[i] for i in nonzero_inds] # permutation according to zeros in z_col   
    z_col = [z_col[i] for i in nonzero_inds] + [z_col[i] for i in zero_inds] # permutation according to zeros in z_col

    # detect zero components in m_vec <-    
     
    equal_inds_list = [] # list of lists where indices of equal sigmas are stored in sublists
    unique_inds_list = [] # indices of unique sigmas (including the first appears of duplicates). 
                          # Needed for construction of permutation matrix
    found_equality_chain = False; curr_equal_inds = []
    prev_val = indexed_sigmas[0]; unique_inds_list.append( prev_val[0] ); i=0; v=0 # def of i,v is needed for correct deletion
    for (i,v) in indexed_sigmas[1:]: # this part of indexed_sigmas is still sorted, other part also thoughs
        if ( v == prev_val[1] ):
            if found_equality_chain == True:
                curr_equal_inds.append( i )    
            else:         
                curr_equal_inds.append( prev_val[0] )
                curr_equal_inds.append( i )
                found_equality_chain = True
        else:
            if found_equality_chain == True:
                equal_inds_list.append( curr_equal_inds )
                unique_inds_list.append(i)                      
                
                curr_equal_inds = []
                found_equality_chain = False
            else:
                unique_inds_list.append(i)             
                
        prev_val = (i,v)
        
    if curr_equal_inds: 
        equal_inds_list.append( curr_equal_inds )
                 
            
    del indexed_sigmas, curr_equal_inds, found_equality_chain, prev_val, i, v
    
    
    equal_sigmas = False # Boolean indicator that there are sigmas which are equal
    if len(equal_inds_list) > 0: # there are equal singular values and we need to do unitary transformation
        U_active = True
        equal_sigmas = True # Boolean indicator that there are sigmas which are equal
        U = np.eye( orig_sigma_len ) # unitary matrix which is a combination of Givens rotations
        
        permute_indices = [] # which indices to put in the end of matrix S and m_col
                             # in m_col this indicas must be zero-valued.
        
        for ll in equal_inds_list:
            U_part = None
            
            m = m_vec[ ll[0] ]
            for i in xrange(1,len(ll)):
                U_tmp = np.eye( len(ll) )
                  
                permute_indices.append( ll[i] )
                
                a = m; b = m_vec[ ll[i] ]
                m = np.sqrt( a**2 + b**2 )
                alpha = a/m; beta = b/m
        
                U_tmp[ 0, 0 ] = alpha; U_tmp[ 0, i ] = beta
                U_tmp[ i, 0 ] = -beta; U_tmp[ i, i ] = alpha
                
                if U_part is None:
                    U_part = U_tmp.copy()
                else:
                    U_part = np.dot( U_tmp, U_part)
                    
            U[ np.array(ll,ndmin=2).T,  np.array(ll,ndmin=2) ] = U_part
       
        extra_indices = permute_indices + zero_inds
        
    else:
        permute_indices = []         
    
    unique_num = len( unique_inds_list )
    equal_num = len( permute_indices )
    assert (orig_sigma_len == unique_num + equal_num + (orig_sigma_len - num_nonzero) ), "Length of equal and/or unique indices is wrong"      
    
    extra_indices = permute_indices + zero_inds
    
    if extra_indices :
        # Permute indices are repeated indices moved to the end of array
        # Sigmas corresponding to permute indices are sorted as well as sigmas corresponding
        # to zero ind. Hence we need to merge these two sigma arrays and take it into 
        # account in the permutation matrix.
        if permute_indices and zero_inds: # need to merge this two arrays
            permute_sigmas =  sigmas[permute_indices]
            zero_sigmas = sigmas[zero_inds]
            
            perm,srt = _arrays_merge( permute_sigmas, zero_sigmas )
            del permute_sigmas, zero_sigmas, srt
            
            permutation_list = unique_inds_list + [ extra_indices[i] for i in perm]
        else: # only one of this lists is nonempty, hence no need to merge
        
            permutation_list = unique_inds_list + extra_indices
        
        
        P = np.zeros( (orig_sigma_len, orig_sigma_len) ) # global permutation matrix
        for (i,s) in enumerate( permutation_list ):
            P[i, s] = 1 
        
        if equal_sigmas:
            U = np.dot(P, U) # incorporate permutation matrix into the matrix U
        else:
            U = P
            
        z_col = np.dot( U, m_vec ) # replace z_col accordingly
        
    else:
        unique_num = orig_sigma_len
        U = None
        z_col = None
    
    if new_col: # return zero value to the last element of sigmas
        sigmas[-1] = 0.0 
        
    return U_active, unique_num, U, z_col


def _arrays_merge(a1, a2, decreasing=True , func=None):
    """
    Auxiliary method which merges two SORTED arrays into one sorted array.
    
    Input:
        a1 - first array
        a2 - second array
        decreasing - a1 and a2 are sorted in decreasing order as well as new array 
        func - function which is used to extract numerical value from the 
               elemets of arrays. If it is None than indexing is used.
        
    Output:
        perm - permutation of indices. Indices of the second array starts 
               from the length of the first array ( len(a1) )
        str -  sorted array
    """
    
    len_a1 = len(a1); len_a2 = len(a2)
    if len_a1 == 0:
        return range(len_a1, len_a1+len_a2), a2
    if len_a2 == 0:
        return range(len_a1), a1
    
    if func is None:
        comp = lambda a,b: (a >= b) if decreasing else (a <= b) # comparison function
    else:
        comp = lambda a,b: ( func(a) >= func(b) ) if decreasing else ( func(a) <= func(b) ) # comparison function
    
    
    inds_a1 = range(len_a1); inds_a2 = range(len_a1, len_a1+len_a2) # indices of array elements
    perm = [np.nan,] * ( len_a1 + len_a2 ) # return permutation array
    srt =  [np.nan,] * ( len_a1 + len_a2 ) # return sorted array
    
    if comp( a1[-1], a2[0]): # already sorted
        srt[0:len_a1] = a1; srt[len_a1:] = a2
        return (inds_a1 + inds_a2), srt
    if comp( a2[-1], a1[0]): # also sorted but a2 goes first
        srt[0:len_a2] = a2; srt[len_a2:] = a1
        return (inds_a2 + inds_a1), srt
    
    a1_ind = 0; a2_ind = 0 # indices of current elements of a1 and a2 arrays
    perm_ind = 0 # current index of output array    
    exit_crit=False
    while (exit_crit==False):
        if comp( a1[a1_ind], a2[a2_ind] ):
            perm[perm_ind] = inds_a1[a1_ind]
            srt[perm_ind] = a1[a1_ind]
            a1_ind += 1
        else:
            perm[perm_ind] = inds_a2[a2_ind]
            srt[perm_ind] = a2[a2_ind]
            a2_ind += 1
        perm_ind += 1
        
        if (a1_ind == len_a1):
            perm[perm_ind:] = inds_a2[a2_ind:]
            srt[perm_ind:] = a2[a2_ind:]
            exit_crit = True
            
        if (a2_ind == len_a2):
            perm[perm_ind:] = inds_a1[a1_ind:]
            srt[perm_ind:] = a1[a1_ind:]
            exit_crit = True

    return perm, srt


def _SVD_upd_diag( sigmas, m_vec, new_col=True):
    """
    This is internal function which is called by update_SVD and SVD_update
    class. It returns the SVD of diagonal matrix augmented by one column. 
    There are two ways to compose augmented matrix. One way is when
    sigma[-1] = 0 and m_vec[-1] != 0 and column m_vec substitutes zero column
    in diagonal matrix np.diag(sigmas). The resulted matrix is square. This
    case is needed when the rank of the original matrix A increased by 1.
    Parameter for this case is new_col=True.    
    The second case is when column m_vec is added to np.diag(sigmas). There
    are no restrictions on value of sigmas and m_vec. This case is used when
    the rank of the of the original matrix A is not increased.
    Parameter for this case is new_col=False.    

    Inputs:
        sigmas - SORTED singular values of a square matrix
        m_vec - extra column added to the square diagonal matrix of singular
                vectors.
        new_col - says which task is task is solved. See comments to the
                  function. True if originally the problem of new column is
                  solved, False if new row is added.
    Outputs:
        U, sigmas, V - SVD of the diagonal matrix plus one column.
        
        !!! Note that unlike update_SVD and scipy SVD routines V 
        not V transpose is returned.
    """
    
    orig_sigmas_length = sigmas.shape[0]
    (equal_sigmas, uniq_sig_num, U_eq, m_vec_transformed) = _SVD_upd_diag_equal_sigmas( sigmas, m_vec, new_col )
    
    if equal_sigmas: # there are equal sigmas
        old_sigmas = sigmas
        old_mvec = m_vec
        
        sigmas = np.diag( np.dot( np.dot( U_eq, np.diag(sigmas) ) , U_eq.T ) )
        extra_sigmas = sigmas[uniq_sig_num:]        
        sigmas = sigmas[0:uniq_sig_num]
        
        m_vec = m_vec_transformed[0:uniq_sig_num]
    
    if (len(sigmas) == 1):
        new_sigmas = np.array( (np.sqrt( sigmas[0]**2 + m_vec[0]**2 ), ) )
        new_size = 1  
    else:
        method = 2
        ret =  find_roots(sigmas, m_vec, method = method)
        new_sigmas = ret
        
        if (method == 3):
            if new_col and  ( (sigmas[-1] < epsilon1) and (sigmas[-2] < epsilon1) ):
                # This check has been written for the case when roots were found by eigh method. So. it should be used
                # when root computing method is 3, if it is 1 this step is done in the function fing_roots.
                # Remind that for the case new_col=True sigmas[-1] = 0 - compulsory.
                # This branch handles the case when there are other zero sigmas.
                new_sigmas[-1] = 0
            
        del ret, method
        
        new_size = len(new_sigmas)
 
    U = np.empty( (new_size, new_size) )
    
    if new_col:
        V = np.empty( (new_size, new_size) )        
    else:
        V = np.empty( (new_size+1, new_size) )
        
    for i in xrange(0, len(new_sigmas) ):
        tmp1 = m_vec / ( (sigmas - new_sigmas[i]) * ( sigmas + new_sigmas[i] ) ) # unnormalized left sv
        if np.any( np.isinf(tmp1) ):
         
            #tmp1[:] = 0
            #tmp1[i] = 1
            if new_sigmas[i] < epsilon1: # new singular value is zero
                tmp1[np.isinf(tmp1)] = 0
            else:
                # we can not determine the value to put instead of infinity. Hence,
                # other property is used to do it. I. e. scalar product of tmp1 and m_vec must equal -1.
                nonzero_inds  = np.nonzero( np.isinf(tmp1) )[0]
                if len( nonzero_inds ) == 1:
                    tmp1[nonzero_inds] = 0
                    tmp1[nonzero_inds] = (-1 - np.dot( tmp1, m_vec)) / m_vec[nonzero_inds]
                else:
                    pass
               
        if np.any( np.isnan(tmp1) ): # temporary check
            pass
        
        nrm = sp.linalg.norm(tmp1, ord=2)
        U[:,i] = tmp1 / nrm
        
        tmp2 =  tmp1 * sigmas# unnormalized right sv
        
        if new_col:
            tmp2[-1] = -1
            nrm = sp.linalg.norm(tmp2, ord=2)
            V[:,i] = tmp2 / nrm
        else:
            #tmp2 = np.concatenate( (tmp2, (-1.0, ) ))
            V[0:-1,i] = tmp2
            V[-1,i] = -1
            nrm = sp.linalg.norm(V[:,i], ord=2)
            V[:,i] = V[:,i] / nrm

    del tmp1, tmp2, nrm
        
    if equal_sigmas:
        extra_sigma_size = orig_sigmas_length -  uniq_sig_num
        eye = np.eye( extra_sigma_size )
        U_eq = U_eq.T
        U = np.dot( U_eq, sp.linalg.block_diag( U, eye ) )
        
        if new_col:
            V = np.dot( U_eq, sp.linalg.block_diag( V, eye ) )
        else:
             V = sp.linalg.block_diag( V, eye )
             P1 = np.eye( orig_sigmas_length, orig_sigmas_length + 1)
             P1 = np.insert(P1, uniq_sig_num, np.array( (0.0,)* ( orig_sigmas_length) + (1.0,) ), axis=0 )
             U_eq = np.hstack( (U_eq, np.array( (0.0,)*U_eq.shape[0], ndmin=2 ).T ) )    
             U_eq = np.vstack( (U_eq, np.array( (0.0,)*U_eq.shape[0] + (1.0,), ndmin=2 ) ) )
             V = np.dot( U_eq, np.dot(P1.T, V))
             
        perm,new_sigmas = _arrays_merge( new_sigmas, extra_sigmas )
                
        new_sigmas = np.array( new_sigmas )
        U = U[:,perm] # replace columns
        V = V[:,perm] # replace columns
    
    return U, new_sigmas, V 


    
def update_SVD( U, S, Vh, a_col, a_col_col=True):
    """
    This is the function which updates SVD decomposition by one column.
    In real situation SVD_updater class is more preferable to use, because
    it is intended for continuous updating and provides some additional 
    features.
    
    Function which updates SVD decomposition of A, when new column a_col is 
    added to the matrix. Actually a_col can be a new row as well. The only
    requirement is that if A has size (m*n) then m >= n. Otherwise, error 
    is raised.
    
    Inputs:
        U,S,Vh - thin SVD of A, which is obtained e.g from scipy.linalg.svd
                S - is a vector of singular values.
                
        a_col - vector with new column (or row of A)
        a_col_col - True if a_col a column, False - if it is row
    
    Outputs:
        U, new_sigmas, Vh - new thin SVD of [A,a_col]
    """

    U_shape = U.shape
    Vh_shape = Vh.shape    
    
    if Vh_shape[0] < Vh_shape[1]:
        raise ValueError("Function update_SVD: Number of columns in V - %i is larger than\
                            the number of rows - %i" % Vh_shape[::-1] )

    if a_col_col and (U_shape[0] != a_col.size):
        raise ValueError("Function update_SVD: Matrix column size - %i and new column size %i\
                            mismatch." % ( U_shape[0],a_col.size) )
                            
    if a_col_col and ( U_shape[0] == Vh_shape[1] ):
        raise ValueError("Function update_SVD:  Column can't be added to the square matrix\
                         set a_col_col=False instead." )                        

    if not a_col_col and (U_shape[1] != a_col.size):
        raise ValueError("Function update_SVD: Matrix row size - %i and new row size %i\
                            mismatch." % ( U_shape[1],a_col.size) )

    zero_epsilon = np.sqrt( np.finfo(np.float64).eps * U.shape[0]**2 * Vh.shape[1] * 10*2)
    
    a_col_old_shape = a_col.shape
    a_col.shape = (a_col.shape[0],) if (len(a_col.shape) == 1) else ( max(a_col.shape), )
    
    if a_col_col: # new column
        old_size = U_shape[1]

        m_vec = np.dot(U.T, a_col) # m vector in the motes    
    
        new_u = a_col - np.dot( U, m_vec) # unnormalized new left eigenvector    
        mu = sp.linalg.norm(new_u, ord=2)
    
              
        Vh = np.hstack( (Vh, np.array( (0.0,)*old_size, ndmin=2 ).T ) )    
        Vh = np.vstack( (Vh, np.array( (0.0,)*old_size + (1.0,), ndmin=2 ) ) )
        
        if (np.abs(mu) < zero_epsilon): # new column is from the same subspace as the old column
                                    # rank is not increased.
            U1, new_sigmas, V1 = _SVD_upd_diag( S, m_vec, new_col=False)
            
        else: 
            U = np.hstack( (U, new_u[:,np.newaxis] / mu) )  
            
            S = np.concatenate( (S, (0.0,) ) )       
            m_vec = np.concatenate((m_vec, (mu,) ))    
        
            U1, new_sigmas, V1 = _SVD_upd_diag( S, m_vec, new_col=True)
            
        U = np.dot(U, U1)
        Vh = np.dot(V1.T,Vh) # V matrix. Need to return V.T though
    
        del U1,V1
    else: # new row
        m_vec = np.dot( Vh, a_col ) 
        
        U = np.vstack( (U, np.array( (0.0,)*U_shape[1], ndmin=2 ) ) )
         
        if (sp.linalg.norm(m_vec, ord=2) < zero_epsilon): # zero row is added
            
            U = U.copy()
            new_sigmas = S.copy()
            Vh = Vh.copy()
        else:
            U = np.hstack( (U, np.array( (0.0,)*U_shape[0] + (1.0,), ndmin=2 ).T ) )  
          
            V1, new_sigmas, U1 = _SVD_upd_diag( S, m_vec, new_col=False) # U1 and V1 are changed because m_vec is new row
        
            U = np.dot(U,U1)
            Vh = np.dot(V1.T, Vh)
        
            del U1,V1
    
    a_col.shape = a_col_old_shape
    
    
    return U, new_sigmas, Vh



def test_root_finder():
    """
    Function which test root finder function.
    
    """    
    times_root_1 = []
    times_root_2 = []    
    times_root_3 = []      
    
    max_diff_1 = []    
    max_diff_2 = []    
    
    sigma_sizes = []
    for k in xrange(0,15):
        
        print k
        sigma_size = 100 + 100 *k
        sigma_sizes.append(sigma_size)
        mult_factor = 100    
        
        sigmas = rnd.random_sample(sigma_size) * mult_factor
        m_vec = rnd.random_sample(sigma_size) * mult_factor
        
        #dk = sp.io.loadmat('root_find.mat')        
        
        #sigmas = dk['sigmas'].squeeze(); m_vec = dk['m_vec'].squeeze(); mu = dk['mu'][0]        
        
        sigmas.sort()
        sigmas = sigmas[::-1]
        with Timer() as t:
            roots1 = find_roots(sigmas, m_vec, method=1)
        times_root_1.append(t.msecs) # roots by root finder        
       
        with Timer() as t:
            roots2 = find_roots(sigmas, m_vec, method=2)
        times_root_2.append(t.msecs) # roots by root finder          
        
        with Timer() as t:
            roots3 = find_roots(sigmas, m_vec, method=3)
        times_root_3.append(t.msecs) # roots by root finder  
        
        maxdiff_ind = np.argmax( np.abs(roots1 - roots3) )
        max_diff_1.append(  np.abs( roots1[maxdiff_ind] - roots3[maxdiff_ind] ) )
        
        maxdiff_ind = np.argmax( np.abs(roots2 - roots3) )
        max_diff_2.append( np.abs( roots2[maxdiff_ind] - roots3[maxdiff_ind] ) )
        
        
    #return np.array( times_root1), np.array( times_root_svd) , np.array(max_diff_roots)
    result = {}
    result['times_root_1'] = times_root_1
    result['times_root_2'] = times_root_2
    result['times_root_3'] = times_root_3
    result['max_diff_1'] = max_diff_1
    result['max_diff_2'] = max_diff_2
    
    io.savemat('root_find.mat', result )
    
    # Plot        
    plt.figure(1)
    plt.plot( sigma_sizes, times_root_3 ,'bo-', label='EIGH root finder')
    plt.plot( sigma_sizes, times_root_2 ,'go-', label='LAPACK inner root finder')
    plt.plot( sigma_sizes,times_root_1 ,'ro-', label='Interval root finder')
    plt.legend(loc=2)    
    plt.title('Computationl Time of 3 root finding methods')    
    plt.xlabel('Matrix size')
    plt.ylabel('Seconds')    
    
    plt.savefig('Root_finder_compute_time.png')
    
    plt.close()
    plt.figure(2)
    plt.plot( sigma_sizes, max_diff_1 ,'bo-')
    plt.title('Max diff. in roots (Interval method and EIGH)')
    plt.ylabel('Difference')
    plt.xlabel('Matrix size')    
    plt.savefig('Root_finder_differences_1.png')    
    
    plt.close()
    plt.figure(3)
    plt.plot( sigma_sizes, max_diff_2 ,'bo-')
    plt.title('Max diff. in roots (LAPACK method and EIGH)')
    plt.xlabel('Matrix size')
    plt.ylabel('Difference')

    plt.savefig('Root_finder_differences_2.png')
    plt.close()

def test_root_finder_lapack():
    """
    """

    sigmas = np.array((6,3,2))
    m_vec = np.array( (3.1, 5.6, 4.5) )
    
    res1 = find_roots(sigmas, m_vec,method=1)
    pass
    res2 = find_roots(sigmas, m_vec,method=2)
    
    return res2
    
def test_update_svd(n_rows,start_n_col, n_max_cols, step_n_col):
    """
    Test SVD update
    """
    
    update_time = []
    new_svd_time = []

    sing_val_diff = []    
    left_sv_diff = []
    right_sv_diff = []
    
    left_orig_ort = []
    left_upd_ort = []
    right_orig_ort = []
    right_upd_ort = []
    
    column_sizes = []; k = 0
    for column_num in xrange(start_n_col,n_max_cols+1,step_n_col):
        k += 1        
        
        print k
       
        column_sizes.append(column_num)
        mult_factor = 1    
        
        matrix = rnd.random_sample((n_rows,column_num-1)) * mult_factor
        new_col = rnd.random_sample(n_rows) * mult_factor    
        
        (um,sm,vm) = sp.linalg.svd(matrix, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)

        with Timer() as t:
            (uu,su,vu) = update_SVD( um, sm, vm, new_col, a_col_col=True)       
        update_time.append(t.msecs/1000.0)
    
        
        with Timer() as t:
            (uf,sf,vf) = sp.linalg.svd( np.hstack( (matrix,new_col[:,np.newaxis] )) , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)      
        new_svd_time.append(t.msecs/1000.0)

        sing_val_diff.append( np.max( np.abs( su - sf) ) )
        left_sv_diff.append( np.max( np.abs(uu) - np.abs(uf) )  )
        right_sv_diff.append( np.max( np.abs(vu) - np.abs(vf) )  )
        
        left_orig_ort.append( np.max( np.abs( np.dot( uf.T, uf) - np.eye( uf.shape[1] ) ) ) )
        left_upd_ort.append( np.max( np.abs( np.dot( uu.T, uu) - np.eye( uu.shape[1] ) ) ) )
        
        right_orig_ort.append( np.max( np.abs( np.dot( vf.T, vf) - np.eye( vf.shape[1] ) ) ) )
        right_upd_ort.append( np.max( np.abs( np.dot( vu.T, vu) - np.eye( vu.shape[1] ) ) ) )
        
        del matrix, um,sm,vm, uu,su,vu, t
        
    result = {}
    result['update_time'] = update_time
    result['new_svd_time'] = new_svd_time
    
    result['sing_val_diff'] = sing_val_diff
    result['left_sv_diff'] = left_sv_diff
    result['right_sv_diff'] = right_sv_diff
    
    result['left_orig_ort'] = left_orig_ort
    result['left_upd_ort'] = left_upd_ort
    result['right_orig_ort'] = right_orig_ort
    result['right_upd_ort'] = right_upd_ort
    
    io.savemat('svd_update.mat', result )   
        
    # Plot        
    plt.figure(1)
    plt.plot( column_sizes, update_time ,'rs-', label='SVD update')
    plt.plot( column_sizes, new_svd_time ,'bo-', label='New SVD')
    plt.legend(loc=2)    
    plt.title('Computational time for SVD update and new SVD.\n Number of matrix rows is %i ' % n_rows )    
    plt.xlabel( 'Number of columns' )
    plt.ylabel('Seconds')    
    
    plt.savefig('SVD_update_time.png')
    
    plt.close()
    plt.figure(2)
    plt.plot( column_sizes,  sing_val_diff ,'bo-')
    plt.title('Max diff. in singular values')
    plt.ylabel('Difference')
    plt.xlabel('Matrix size')    
    plt.savefig('SVD_update_sing_diff.png')    
    
    plt.close()
    plt.figure(3)
    plt.plot( column_sizes,  left_sv_diff ,'bo-')
    plt.title('Max diff. in left singular vectors')
    plt.ylabel('Difference')
    plt.xlabel('Matrix size')    
    plt.savefig('SVD_update_left_sv_diff.png')
    
    plt.close()
    plt.figure(4)
    plt.plot( column_sizes,  right_sv_diff ,'bo-')
    plt.title('Max diff. in right singular vectors')
    plt.ylabel('Difference')
    plt.xlabel('Matrix size')    
    plt.savefig('SVD_update_right_sv_diff.png')
    
    plt.close()
    plt.figure(5)
    plt.plot( column_sizes,  left_orig_ort ,'bo-', label='new SVD')
    plt.plot( column_sizes,  left_upd_ort ,'ro-', label='SVD update')
    plt.title('Max diff. in left sv orthogonality')
    plt.ylabel('Difference')
    plt.xlabel('Matrix size')
    plt.legend(loc=2)      
    plt.savefig('SVD_update_left_sv_ortog_diff.png')
    
    plt.close()
    plt.figure(5)
    plt.plot( column_sizes,  right_orig_ort ,'bo-', label='new SVD')
    plt.plot( column_sizes,  right_upd_ort ,'ro-', label='SVD update')
    plt.title('Max diff. in right sv orthogonality')
    plt.ylabel('Difference')
    plt.xlabel('Matrix size')
    plt.legend(loc=2)      
    plt.savefig('SVD_update_right_sv_ortog_diff.png')
    
    plt.close()
    
def test_array_merge():
    a1 = np.array([5,4])
    a2 = np.array([6,2,1])
    
    res = _arrays_merge(a1,a2)
    
    return res
    
def test_equal_sigmas():
    import numpy.random as rnd    
    
    #S = np.array( [5, 4, 2.34, 2.34, 1.5, 1.34, 1.34, 1.34, 0.0] )
    #S = np.array( [3, 3, 3, 3, 3, 2, 2, 0.0, 0.0 ] )
    #S = np.array( [3, 3, 2, 0.0 ] )
    #m_col = np.array( [0.0, 3.23, 4.21, 0.0, 5, 32, 11,12, 13] )
    #m_col = np.array( [5, 2, 0.0, 0.0, 0.0, 0, 0,0.0, 0.0] )
    # m_col = np.array( [5, 2, 0.0, 4] )
    #m_col = np.array( [3, 3, 3, 3, 3, 3, 3, 3, 3] )
    
    vector_length = 9 # actual number is this number plus one.
    
    var_dict = {}
    iteration_no = 0
    for tt in range(0,500):
        for no_different_sigmas in range(1,10):
            #no_different_sigmas = 3
            for no_zeros_in_m_col in range(0,10):
            # no_zeros_in_m_col = 3
                
                iteration_no += 1
                print iteration_no
            
                available_sigmas = np.arange( vector_length )
                S = sorted( rnd.choice( rnd.choice( available_sigmas, size=no_different_sigmas ), size=vector_length), reverse=True )
                S = np.array(S + [0,])
                #S = np.array( [6,6,6,6,6,6,0,0,0,0.0 ])
                m_col = np.round( vector_length*rnd.rand( vector_length) - vector_length / 2.0, 2 )
                m_col[ rnd.choice( np.arange(vector_length), size = no_zeros_in_m_col, replace=False ) ] = 0.0
                m_col = np.concatenate( (m_col, rnd.rand(1)*vector_length - vector_length / 2.0 ) )    
                
                M = np.hstack( ( np.vstack( (np.diag(S[0:-1]), np.zeros( (1,len(m_col) - 1) ) ) ), m_col[:, np.newaxis ]))
                #M = np.hstack( (np.diag(S), m_col[:, np.newaxis ]))    
                
                (U,Ss,V) = _SVD_upd_diag( S, m_col, new_col=True  )
                M1 = np.dot( np.dot(U, np.diag(Ss)), V.T )
                if np.max( np.abs( M - M1) )/S[0] > 10**(-12):
                    var_dict[ "m_col_%i" % iteration_no ] = m_col
                    var_dict[ "S_%i" % iteration_no ] = S
                    print "Col add, iteration %i:  %e" % (iteration_no, ( np.max( np.abs( M - M1) )/S[0]) )                 
                    
                if np.sum( np.abs( m_col[0:-1] )) > 0 and  np.sum( np.abs( S[0:-1] )) > 0:
                    M = np.hstack( (np.diag( S[0:-1] ), m_col[0:-1, np.newaxis ]))
                    (U,Ss,V) = _SVD_upd_diag( S[0:-1], m_col[0:-1], new_col=False  )
                    M1 = np.dot( np.dot(U, np.diag(Ss)), V.T )
                    if np.max( np.abs( M - M1) )/S[0] > 10**(-12):
                        var_dict[ "m_col_%i" % iteration_no ] = m_col
                        var_dict[ "S_%i" % iteration_no ] = S   
                        print "Row add, iteration %i:  %e" % (iteration_no, ( np.max( np.abs( M - M1) )/S[0]) )
                
                
    if var_dict:
        io.savemat('Test_SVD_upd_1.mat', var_dict )
    
    #(uf,sf,vf) = sp.linalg.svd( M , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
    #M2 = np.dot( np.dot(uf, np.diag( sf )), vf )
    
    return True
    
def test_update_diag( sigmas, m_vec, new_col=True,U=None,S=None,V=None):
    #dct = io.loadmat( 'Fri_file' )
    #sigmas = dct['sigmas']
    #m_vec = dct['m_vec']
    
    if new_col:
        M = np.hstack( ( np.vstack( (np.diag(sigmas[0:-1]), np.zeros( (1,len(m_vec) - 1) ) ) ), m_vec[:, np.newaxis ]))
    else:
        M = np.hstack( (np.diag( S ), m_vec[:, np.newaxis ]))
        
    (Um,Sm,Vm) = sp.linalg.svd(M, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
    Ar1 = np.dot( Um, np.dot(np.diag(Sm), Vm ) )
    diff1 = np.max( np.abs( M - Ar1) )/Sm[0]
    
    if not U is None:
        (U, S, V) = _SVD_upd_diag(sigmas, m_vec, new_col)
        
    Ar2 = np.dot( U, np.dot(np.diag(S), V.T ) )
    Ar = Ar1 - Ar2
    diff2 = np.max( np.abs( M - Ar2) )/S[0]

    
    return diff2, Um,Sm,Vm

def test_SVD_comp_complexity(n_rows,start_n_col, n_max_cols, step_n_col):
    """
    Determine complexity of standard SVD computation
    """
    
    A = rnd.rand(n_rows, start_n_col)
    #(U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
     
    #svd_comp = np.empty((n_max_cols, 9))
    times = np.empty((n_max_cols, 2))    
    
    iter_counter = 0
    for ii in xrange( 0,n_max_cols,step_n_col ):
       
        a1 = rnd.rand(n_rows,step_n_col)
          
        A = np.hstack( (A,a1) )
        B = A.T        
        
        with Timer() as t:
            (Us,Ss,Vhs) = sp.linalg.svd( B , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
            
        times[iter_counter,0] = A.shape[1]
        times[iter_counter,1] = t.msecs
        iter_counter += 1
        
        print ii
    
    times = times[0:iter_counter,:]
    return times
    
if __name__ == '__main__':
    
    #test_root_finder()
    
    #test_root_finder()

    #test_update_svd()

    #res = test_array_merge()
    
    #res = test_equal_sigmas()

    #test_SVD_updater()


    res = test_SVD_update_reorth(1000,1000, 1500, 0, 50) # (n_rows,start_n_col, n_max_cols, prob_same_subspace, reorth_step)
    
    #res = test_SVD_comp_complexity(10000,10, 13000, 500)
    
    #test_root_finder_lapack()
    
    #test_root_finder()
    
    #test_update_svd(10000,500, 6000, 500)
    
    #sigmas = np.array( [30000, 0.5, 0])
    #m_vec = np.array( [0.001, -100, 10000])
    #test_update_diag(sigmas,m_vec)