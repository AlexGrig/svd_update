# -*- coding: utf-8 -*-
"""
Testing functions from SVD update functionality.
"""
import numpy as np
import scipy as sp
import numpy.random as rnd
import scipy.io as io

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from svd_updater import *

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
#    SVD_upd = SVD_updater( U,S,Vh, update_V = True, reorth_step=10)
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


    #Test column update. Thin matrix, rank not increases
    n_rows = 1000; n_cols = 800
    A = rnd.rand(n_rows,n_cols)
    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )

    ##a1 = rnd.rand(5,1)
    a1 = np.dot(A, rnd.rand(n_cols,1) )
    A = np.hstack( (A,a1) )
    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )

    SVD_upd = SVD_updater( U,S,Vh, update_V = True, reorth_step=10)

    (Us1, Ss1, Vhs1) = SVD_upd.add_column(a1 )
    Ar1 = np.dot( Us1, np.dot(np.diag(Ss1), Vhs1 ) )

    diff1 = np.max( np.abs( A - Ar1) )/St[0]

    a2 = rnd.rand(n_rows,1)
    #a2 = np.dot(A, np.array([2,1,4,-3],ndmin=2 ).T )
    A = np.hstack( (A,a2) )
    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
    (Us2, Ss2, Vhs2) = SVD_upd.add_column( a2 )

    Ar2 = np.dot( Us2, np.dot(np.diag(Ss2), Vhs2 ) )

    diff2 = np.max( np.abs( A - Ar2) )/St[0]

    return diff2

#    # Test column update. Fat matrix
#    A = rnd.rand(5,4)
#    A = rnd.rand(5,5)
#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#
#    a1 = rnd.rand(5,1)
#    #a1 = np.dot(A, np.array([2,1,4],ndmin=2 ).T )
#    A = np.hstack( (A,a1) )
#    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#
#    SVD_upd = SVD_updater( U,S,Vh, update_V = True, reorth_step=10)
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

    # test function update_SVD
#    A = rnd.rand(5,3)
#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True,
#                             overwrite_a=False, check_finite=False )
#
#    #a1 = rnd.rand(5,1)
#    a1 = np.dot(A, np.array([2,1,4],ndmin=2 ).T )
#    A = np.hstack( (A,a1) )
#    (Ut,St,Vht) = sp.linalg.svd( A  , full_matrices=False, compute_uv=True,
#                                      overwrite_a=False, check_finite=False )
#
#    (Us1, Ss1, Vhs1) = update_SVD( U, S, Vh, a1, a_col_col=True)
#    Ar1 = np.dot( Us1, np.dot(np.diag(Ss1), Vhs1 ) )
#
#    diff1 = np.max( np.abs( A - Ar1) )/St[0]
#
#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True,
#                                  overwrite_a=False, check_finite=False )
#
#    #a2 = np.array([2,1,4,7],ndmin=2 )
#    a2 = np.array([0,0,0,0],ndmin=2 )
#    A = np.vstack( (A,a2) )
#
#    (Us2, Ss2, Vhs2) = update_SVD( U, S, Vh, a2, a_col_col=False)
#    Ar2 = np.dot( Us2, np.dot(np.diag(Ss2), Vhs2 ) )
#
#    diff2 = np.max( np.abs( A - Ar2) )/St[0]
#
#    return diff1

def test_SVD_update_reorth(n_rows,start_n_col, n_max_cols, prob_same_subspace,
                           reorth_step,file_name):
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

    test_update_SVD_function = False
    #n_rows = 1000
    #n_max_cols = 1000

#    (X,Y) = ds.make_regression(n_samples = 1000, n_features = 7,n_informative=5, \
#                       n_targets=2, bias = 2.0, effective_rank = 2)
    A = rnd.rand(n_rows, start_n_col)
    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )

    svd_upd = SVD_updater( U,S,Vh, update_V = True, reorth_step=reorth_step)

    if test_update_SVD_function:
        svd_comp = np.empty((n_max_cols, 10))
        times = np.empty((n_max_cols, 5))
    else:
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

        if test_update_SVD_function:
            with Timer() as t:
                (U, S, Vh) = update_SVD( U, S, Vh, a1, a_col_col=True)
            times[ii,4] = t.msecs

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

        svd_comp[ii,6] =  np.max( np.abs( Ar1 - A )/Ss[0] )

        svd_comp[ii,7] =  np.max( np.abs(np.dot( Ut.T, Ut ) - np.eye( Ut.shape[1]) ) )
        svd_comp[ii,8] = np.max( np.abs(np.dot( Vht, Vht.T ) - np.eye( Vht.shape[0]) ) )
        if test_update_SVD_function:
            try:
                tmp = np.abs( Ss[0:S.shape[0]] - S ) / Ss[ 0:S.shape[0] ]
            except ValueError as e:
                raise e
            svd_comp[ii,9] = np.max( tmp )
        print ii

#    (U,S,Vh) = sp.linalg.svd( A , full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False )
#    Ar2 = np.dot( U, np.dot(np.diag(S), Vh ) )

    t1 = svd_comp; t2 = times

    result = {}
    result['svd_comp'] = svd_comp
    result['times'] = times
    result['n_rows'] = n_rows

    #io.savemat('new_inc_svd_update_1000_10_0d05_reorth.mat', result )
    io.savemat(file_name, result )

#    # Plot
#    plt.rc('xtick', labelsize=18)
#    plt.rc('ytick', labelsize=18)
#    plt.figure(1)
#    plt.plot( t1[:,4], t2[:,3]/1000.0, 'ko-', label='New SVD')
#    plt.plot( t1[:,4], t2[:,2]/1000.0, 'rs-', label='SVD update')
#    plt.plot( t1[:,4], t2[:,0]/1000.0, 'm--',
#              label='SVD update: except matrix mult.')
#    plt.plot( t1[:,4], t2[:,1]/1000.0, 'y-.',
#              label='SVD update: matrix mult.')
#    plt.legend(loc=2,prop={'size':16})
#    #plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
#    #plt.suptitle('Comp. time of sequential SVD update, its components and new SVD', fontsize=20)
#    plt.xlabel( 'Number of columns' , fontsize = 17 )
#    plt.ylabel( 'Seconds' , fontsize = 17 )
#    plt.xlim( (0,1000) )
#    plt.ylim( (0,1.2) )
#    plt.show()
    #plt.savefig('Seq_SVD_update_2.png')
#
#    plt.close()

#    sv_diffs = None
#    for ii in range(1,11):
#        dct = io.loadmat( '1000_1000_no_reorth_%i.mat' % ii )
#        t1 = dct['svd_comp']
#        t2 = dct['times']
#
#        new_col = t1[:,0]
#        new_col.shape = ( new_col.shape[0],1 )
#        if ii == 1:
#            sv_diffs = new_col
#        else:
#            sv_diffs = np.hstack( ( sv_diffs, new_col ) )
#
#    pass
#    sv_mean = np.mean( sv_diffs, axis=1)
#    sv_std = np.std( sv_diffs, axis=1)
#    del new_col, ii, t2
    sv_mean = t1[:,0]
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    fig = plt.figure(2)
    ax1 = fig.add_subplot(1,1,1)

    line_1 = ax1.plot( t1[:,4], sv_mean, 'ko-', label='(Left Y Axis) Max. Singular Value Difference')
    #line_2 = ax1.plot( t1[:,4], sv_mean + 2* sv_std, 'k--')
    #line_3 = ax1.plot( t1[:,4], sv_mean - 2* sv_std, 'k--', label='(Left Y Axis) Two Standard Deviations of Sing. Val.')

    ax1.set_ylabel('Max. Singular Value Difference', fontsize = 18 )
    #ax1.set_ylim( 0,0.25*10**(-7) ) # 1000_10
    ax1.set_ylim( 0,0.5*10**(-8) ) # 1000_1000
    #plt.suptitle('Squential SVD. Maximum relative singular value differences',
    #             fontsize=20)
    #plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
    ax2 = ax1.twinx()
    #ax2.set_ylim( 0,2*10**(-11) ) # 1000_10
    ax2.set_ylim( 0,1.2*10**(-10) ) # 1000_1000
    line_2 = ax2.plot( t1[:,4], t1[:,6], 'ys-', label='(Right Y Axis) Max. Reconstruction Difference' )
    ax2.set_ylabel( 'Max. Reconstruction Difference', fontsize = 18 )
    ax1.set_xlabel('Number of columns', fontsize = 18 )

    # added these three lines
    line_1, label_1 = ax1.get_legend_handles_labels()
    line_2, label_2 = ax2.get_legend_handles_labels()
    plt.legend( line_1 + line_2, label_1 + label_2, loc=2 )
    #plt.savefig('Seq_SVD_sing_val_diff_2.png')
    plt.xlim( (0,1000) )
    #plt.xlim( (1000,2500) )
    plt.show()
    plt.close()

#    plt.rc('xtick', labelsize=18)
#    plt.rc('ytick', labelsize=18)
#    plt.figure(3)
#    plt.plot(t1[:,4], t1[:,4], 'ko-',
#              label='Total Number of Singular Values')
#    plt.plot(t1[:,4], t1[:,1], 'rs-',
#              label='Index of Maximum Singular Value ifference')
#    plt.legend(loc=2, prop={'size':16})
#    #plt.suptitle('Squential SVD. Index of maximum difference singular value',
#    #              fontsize = 20 )
#    #plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17)
#    plt.ylabel('Index', fontsize = 18 )
#    plt.xlabel('Number of columns', fontsize = 18)
#    #plt.xlim( (0,1000) )
#    plt.xlim( (1000,2500) )
#    plt.ylim((0,2800))
#    plt.show()

#    return (svd_comp,times)
#
    # Other plots
    xx = [2999, 4752, 6344, 13760, 30487]
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.figure(3)
    plt.plot( xx, [ 15.83, 23.94, 32.87, 94.94, 225.46 ], 'ko-', linewidth=3, ms=10, \
              label='OP-ELM' )
    plt.plot( xx, [ 10.09, 14.77, 14.53, 66.67, 163.33 ], 'rs-', linewidth=3, ms=10, \
              label='(Inc)-OP-ELM' )
    #plt.yscale('log')
    plt.legend( loc=2,prop={'size':18} )
    #plt.suptitle('Running Time with Respect to Number of Samples',
    #              fontsize = 20)
    #plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
    plt.ylabel( 'Seconds', fontsize = 18 )
    plt.xlabel( 'Number of Training Samples', fontsize = 18 )
    plt.show()
#
#    xx = [100,300,600]
#    plt.rc('xtick', labelsize=18)
#    plt.rc('ytick', labelsize=18)
#    plt.figure(3)
#    plt.plot(xx, [1.64, 15.83, 65.41], 'ko-', linewidth=3, ms=10,
#             label='OP-ELM')
#    plt.plot(xx, [1.33, 10.09, 38.96], 'rs-', linewidth=3, ms=10,
#             label='(Inc)-OP-ELM')
#    plt.legend(loc=2,prop={'size':18})
#    #plt.suptitle('Running Time with Respect to Initial Number of Neurons',
#    #             fontsize = 20 )
#    #plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
#    plt.ylabel('Seconds', fontsize = 18 )
#    plt.xlabel('Initial Number of Neurons', fontsize = 18 )
#    plt.show()


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
    n_rows = 10000 #10000,500, 6000, 500
    column_sizes = range(500,6001,500)

    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.figure(1)
    plt.plot( column_sizes, new_svd_time ,'ko-', label='New SVD')
    plt.plot( column_sizes, update_time ,'rs-', label='SVD update')
    plt.legend(loc=2)
    #plt.title('Computational time for SVD update and new SVD.\n Number of matrix rows is %i ' % n_rows )
    plt.xlabel( 'Number of columns', fontsize = 17)
    plt.ylabel('Seconds', fontsize = 17)
    plt.show()


    plt.figure(1)
    plt.plot( t1[:,4], t2[:,3]/1000.0, 'ko-', label='New SVD')
    plt.plot( t1[:,4], t2[:,2]/1000.0, 'rs-', label='SVD update')
    plt.plot( t1[:,4], t2[:,0]/1000.0, 'm--',
              label='SVD update: except matrix mult.')
    plt.plot( t1[:,4], t2[:,1]/1000.0, 'y-.',
              label='SVD update: matrix mult.')
    plt.legend(loc=2,prop={'size':16})
    #plt.title('Number of matrix rows is %i ' % n_rows, fontsize = 17 )
    #plt.suptitle('Comp. time of sequential SVD update, its components and new SVD', fontsize=20)
    plt.xlabel( 'Number of columns' , fontsize = 17 )
    plt.ylabel( 'Seconds' , fontsize = 17 )
    plt.show()

    #plt.savefig('SVD_update_time.png')

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

def test_update_diag(sigmas, m_vec, new_col=True,U=None,S=None,V=None):
    #dct = io.loadmat( 'Fri_file' )
    #sigmas = dct['sigmas']
    #m_vec = dct['m_vec']

    if new_col:
        M = np.hstack((np.vstack((np.diag(sigmas[0:-1]), np.zeros((1,len(m_vec) - 1)))), m_vec[:, np.newaxis ]))
    else:
        M = np.hstack((np.diag(S), m_vec[:, np.newaxis]))

    (Um,Sm,Vm) = sp.linalg.svd(M, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
    Ar1 = np.dot( Um, np.dot(np.diag(Sm), Vm ) )
    diff1 = np.max( np.abs( M - Ar1) )/Sm[0]

    if not U is None:
        (U, S, V) = _SVD_upd_diag(sigmas, m_vec, new_col)

    Ar2 = np.dot(U, np.dot(np.diag(S), V.T))
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

    #test_update_svd()

    #res = test_array_merge()

    #res = test_equal_sigmas()

    #test_SVD_updater()


#    res = test_SVD_update_reorth(1000,10, 990, 0.05, 2000,'1000_10_no_reorth_1.mat') # (n_rows,start_n_col, n_max_cols, prob_same_subspace, reorth_step)
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_1.mat') # (n_rows,start_n_col, n_max_cols, prob_same_subspace, reorth_step)
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_2.mat')
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_3.mat')
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_4.mat')
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_5.mat')
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_6.mat')
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_7.mat')
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_8.mat')
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_9.mat')
#    res = test_SVD_update_reorth(1000, 1000, 1500, 0.05, 2000, '1000_1000_no_reorth_10.mat')
    
    
    #res = test_SVD_update_reorth(1000,10, 990, 0.05, 50) # (n_rows,start_n_col, n_max_cols, prob_same_subspace, reorth_step)
    #res = test_SVD_comp_complexity(10000,10, 13000, 500)

    #test_root_finder_lapack()

    #test_root_finder()

    #test_update_svd(10000,500, 6000, 500)

    #sigmas = np.array( [30000, 0.5, 0])
    #m_vec = np.array( [0.001, -100, 10000])
    #test_update_diag(sigmas,m_vec)
    sigmas = np.array([4., 3., 2., 0])
    m_vec = np.array([3.12, 5.7, -4.8, -2.2])
    
    # Test SVD ->
    M = np.hstack((np.vstack((np.diag(sigmas[0:-1]), np.zeros((1,len(m_vec) - 1)))), m_vec[:, np.newaxis ]))
    SM = sp.linalg.svd(M, full_matrices=False, compute_uv=False, overwrite_a=False, check_finite=False )
    # Test SVD <-
    
    it_len = len(sigmas)
    sgm = np.concatenate((sigmas[::-1], (sigmas[0] + it_len*np.sqrt(np.sum( np.power(m_vec,2))),)))
    mvc = np.concatenate((m_vec[::-1], (0,)))
    roots = []

    it_start = 1
    prepend_zero = False

    for i in xrange(it_start, it_len+1): # find all singular except the last
        res = sp.linalg.lapack.dlasd4(i, sgm, mvc)
        roots.append(res[1])

        if  (res[3] > 0) or np.any(np.isnan(roots)):
            raise ValueError("LAPACK root finding dlasd4 failed to fine the last singular value")
    
    pass
    
    
