def MWT_Apply_Operator(data, operator, time_indices=None, block_size=4096):
    """Apply a cached exact discrete operator built by supplied MWT(identity).

    operator must come from MWT(np.eye(N), j0,j1,axis=0,boundary=...,
    normalization='physical',return_reconstruction=False). It has shape
    (3,N,N): output scale/time by original input time. This function does not
    replace MWT with another basis. Cache source SHA, j0/j1, boundary and norm.
    All N contiguous original input times are required even when only specified
    output time_indices are returned. Invalid whole time columns remain NaN.
    Data may be an array (time,...) or a dictionary of arrays with the same N.
    Return shape is (3,number_selected_times,...), same coefficient meaning.
    No reconstructions, derivatives, I/O, or external custom-code dependencies.
    """
    import numpy as np
    op=np.asarray(operator,dtype=float)
    if op.ndim!=3 or op.shape[0]!=3 or op.shape[1]!=op.shape[2] or not np.isfinite(op).all(): raise ValueError('operator must be finite (3,N,N)')
    n=op.shape[-1]
    if n<4 or n&(n-1): raise ValueError('operator N must be a power of two')
    if isinstance(block_size,bool) or int(block_size)!=block_size or block_size<=0: raise ValueError('block_size must be positive integer')
    if time_indices is None: ix=np.arange(n)
    else:
        ix=np.asarray(time_indices)
        if ix.ndim!=1 or not np.issubdtype(ix.dtype,np.integer) or not len(ix) or np.any(ix<0) or np.any(ix>=n): raise ValueError('invalid output time indices')
        if len(np.unique(ix))!=len(ix): raise ValueError('duplicate output time indices')
    op_selected=np.ascontiguousarray(op[:,ix,:].reshape(3*len(ix),n))
    def apply(x):
        if np.iscomplexobj(x): raise TypeError('real input required')
        a=np.asarray(np.ma.asarray(x,dtype=float).filled(np.nan))
        if not a.ndim or a.shape[0]!=n or np.isinf(a).any(): raise ValueError('input time length/finite mismatch')
        f=a.reshape(n,-1);out=np.full((3*len(ix),f.shape[1]),np.nan,dtype=float)
        for start in range(0,f.shape[1],int(block_size)):
            block=f[:,start:start+int(block_size)]
            good=np.isfinite(block).all(axis=0)
            ids=np.flatnonzero(good)+start
            if len(ids):out[:,ids]=op_selected@block[:,good]
        return out.reshape((3,len(ix))+a.shape[1:])
    if isinstance(data,dict):return {k:apply(v) for k,v in data.items()}
    return apply(data)
