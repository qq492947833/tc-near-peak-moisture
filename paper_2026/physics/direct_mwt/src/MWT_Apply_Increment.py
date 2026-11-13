def MWT_Apply_Increment(data, operator, time_indices, cached_coefficients=None,
                        cached_time_indices=None, cached_spatial_mask=None,
                        block_size=1024):
    """Apply the original exact MWT operator only where coefficients are absent.

    data: (N,pressure,latitude,longitude), original field or original product.
    operator: unchanged physical-normalization (3,N,N) matrix.
    cached_coefficients: (3,n_cached,pressure,latitude,longitude), on the output
    grid, with original values only where cached_spatial_mask is True.
    cached_time_indices: output indices relative to this scheme's original N.
    The spatial mask is 2-D and marks original cache columns, including their
    original NaNs; those cells are copied exactly, never transformed again.
    Returns coefficients and transform/copy counts, no file/global dependency.
    """
    import numpy as np
    a = np.asarray(data, dtype=float)
    op = np.asarray(operator, dtype=float)
    indices = np.asarray(time_indices)
    if a.ndim != 4 or op.shape != (3, a.shape[0], a.shape[0]) or not np.isfinite(op).all():
        raise ValueError('Input/operator dimensions differ')
    if a.shape[0] < 4 or a.shape[0] & (a.shape[0] - 1) or np.isinf(a).any():
        raise ValueError('Invalid original time series')
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer) or len(set(indices)) != len(indices):
        raise ValueError('Output times must be distinct integer operator rows')
    if indices.min() < 0 or indices.max() >= a.shape[0] or int(block_size) != block_size or block_size <= 0:
        raise ValueError('Output rows/block size out of range')
    output = np.full((3, len(indices)) + a.shape[1:], np.nan)
    existing = {}
    mask = np.zeros(a.shape[-2:], bool)
    if cached_coefficients is not None:
        cached = np.asarray(cached_coefficients, dtype=float)
        cached_indices = np.asarray(cached_time_indices)
        mask = np.asarray(cached_spatial_mask)
        if mask.shape != a.shape[-2:] or mask.dtype != np.bool_:
            raise ValueError('Cached spatial coverage must be a 2-D Boolean grid mask')
        if cached.shape != (3, len(cached_indices)) + a.shape[1:] or np.isinf(cached).any():
            raise ValueError('Cached coefficient dimensions/values differ')
        if not np.issubdtype(cached_indices.dtype, np.integer) or len(set(cached_indices)) != len(cached_indices):
            raise ValueError('Cached output rows are not distinct integers')
        existing = {int(t): i for i, t in enumerate(cached_indices)}
        for i, t in enumerate(indices):
            if int(t) in existing:
                output[:, i, :, mask] = cached[:, existing[int(t)], :, mask]
    elif cached_time_indices is not None or cached_spatial_mask is not None:
        raise ValueError('Partial cache arguments')
    flat = a.reshape(a.shape[0], -1)
    outflat = output.reshape(3, len(indices), -1)
    spatial = np.broadcast_to(mask, a.shape[1:]).reshape(-1)
    groups = [([i for i, t in enumerate(indices) if int(t) not in existing], np.ones(flat.shape[1], bool)),
              ([i for i, t in enumerate(indices) if int(t) in existing], ~spatial)]
    transformed_columns_times = 0
    for positions, allowed in groups:
        if not positions or not allowed.any():
            continue
        matrix = np.ascontiguousarray(op[:, indices[positions], :].reshape(3 * len(positions), a.shape[0]))
        for start in range(0, flat.shape[1], int(block_size)):
            block = flat[:, start:start + int(block_size)]
            good = np.isfinite(block).all(axis=0) & allowed[start:start + block.shape[1]]
            cols = np.flatnonzero(good) + start
            if not len(cols):
                continue
            values = (matrix @ block[:, good]).reshape(3, len(positions), len(cols))
            for j, output_position in enumerate(positions):
                outflat[:, output_position, cols] = values[:, j, :]
            transformed_columns_times += len(cols) * len(positions)
    copied = sum(int(t) in existing for t in indices) * int(spatial.sum())
    return output, {'copied_column_times': copied,
                    'transformed_valid_column_times': int(transformed_columns_times),
                    'new_output_time_rows': int(sum(int(t) not in existing for t in indices)),
                    'existing_output_time_rows': int(sum(int(t) in existing for t in indices))}
