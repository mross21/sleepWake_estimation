#%%
import pandas as pd
import numpy as np

file = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/BiAffect_data/processed_output/keypress/User_94_keypressDataMetrics.csv'
pathOut = '/home/mindy/Desktop/BiAffect-iOS/UnMASCK/graph_regularized_SVD/'
df = pd.read_csv(file, index_col=False)

#%%

df['hour'] = pd.to_datetime(df['keypressTimestampLocal']).dt.hour

M = df.groupby(['dayNumber','hour'],as_index = False).size().pivot('hour','dayNumber').fillna(0)
# rows = hour
# columns = day number
display(M)

# M.to_csv(pathOut+'matrix.csv', index=False, header=False)

# %%
def regularized_svd(X, B, rank, alpha, as_sparse=False):
    """
    Perform graph regularized SVD as defined in
    Vidar & Alvindia (2013).

    Parameters
    ----------
    X : numpy array
        m x n data matrix.

    B : numpy array
        n x n graph Laplacian of nearest neighborhood graph of data.

    W : numpy array
        n x n weighted adjacency matrix.

    rank : int
        Rank of matrix to approximate.

    alpha : float
        Scaling factor.

    as_sparse : bool
        If True, use sparse matrix operations. Default is False.

    Returns
    -------
    H_star : numpy array
        m x r matrix (Eq 15).

    W_star : numpy array
        r x n matrix (Eq 15).
    """
    import numpy as np
    from scipy.linalg import svd
    from scipy.linalg import cholesky
    from scipy.linalg import inv
    import scipy.sparse as sp
    from sklearn.utils.extmath import randomized_svd
    from sksparse import cholmod

    if as_sparse:
        # Use sparse matrix operations to reduce memory
        I = sp.lil_matrix(B.shape)
        I.setdiag(1)
        C = I + (alpha * B)
        print('Computing Cholesky decomposition')
        factor = cholmod.cholesky(C)
        D = factor.L()
        print('Computing inverse of D.T')
        invDt = sp.linalg.inv(D.T)
        # Eq 11
        print('Computing randomized SVD')
        E, S, Fh = randomized_svd(X @ invDt,
                                  n_components=rank,
                                  random_state=123)
        E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
        H_star = E_tilde  # Eq 15
        W_star = E_tilde.T @ X @ sp.linalg.inv(C)  # Eq 15

    else:
        # Eq 11
        I = np.eye(B.shape[0])
        C = I + (alpha * B)
        D = cholesky(C)
        E, S, Fh = svd(X @ inv(D.T))
        E_tilde = E[:, :rank]  # rank-r approximation; H_star = E_tilde (Eq 15)
        H_star = E_tilde  # Eq 15
        W_star = E_tilde.T @ X @ inv(C)  # Eq 15
    return H_star, W_star