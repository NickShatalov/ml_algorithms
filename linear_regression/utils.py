def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    return np.apply_along_axis(
        lambda e: (function(*(w + eps * e)) - function(*w)) / eps,
        0,
        np.eye(w.size)
    )
