def diff(x, axis):
    """Take the finite difference of a tensor along an axis.
    Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.
    Returns:
    d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
    ValueError: Axis out of range for tensor.
    """
    shape = x.shape

    begin_back = [0 for unused_s in range(len(shape))]
#     print("begin_back",begin_back)
    begin_front = [0 for unused_s in range(len(shape))]

    begin_front[axis] = 1
#     print("begin_front",begin_front)

    size = list(shape)
    size[axis] -= 1
#     print("size",size)
    slice_front = x[begin_front[0]:begin_front[0]+size[0], begin_front[1]:begin_front[1]+size[1]]
    slice_back = x[begin_back[0]:begin_back[0]+size[0], begin_back[1]:begin_back[1]+size[1]]

#     slice_front = tf.slice(x, begin_front, size)
#     slice_back = tf.slice(x, begin_back, size)
#     print("slice_front",slice_front)
#     print(slice_front.shape)
#     print("slice_back",slice_back)

    d = slice_front - slice_back
    return d


def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.
    Returns:
    unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
    #     print("dd",dd)
    ddmod = np.mod(dd + np.pi, 2.0 * np.pi) - np.pi  # ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
    #     print("ddmod",ddmod)

    idx = np.logical_and(np.equal(ddmod, -np.pi),
                         np.greater(dd, 0))  # idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
    #     print("idx",idx)
    ddmod = np.where(idx, np.ones_like(ddmod) * np.pi,
                     ddmod)  # ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
    #     print("ddmod",ddmod)
    ph_correct = ddmod - dd
    #     print("ph_corrct",ph_correct)

    idx = np.less(np.abs(dd), discont)  # idx = tf.less(tf.abs(dd), discont)

    ddmod = np.where(idx, np.zeros_like(ddmod), dd)  # ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = np.cumsum(ph_correct, axis=axis)  # ph_cumsum = tf.cumsum(ph_correct, axis=axis)
    #     print("idx",idx)
    #     print("ddmod",ddmod)
    #     print("ph_cumsum",ph_cumsum)

    shape = np.array(p.shape)  # shape = p.get_shape().as_list()

    shape[axis] = 1
    ph_cumsum = np.concatenate([np.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    # ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
    #     print("unwrapped",unwrapped)
    return unwrapped
