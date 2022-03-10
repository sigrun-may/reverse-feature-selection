from group-lasso import GroupLasso

gl = GroupLasso(
    groups=groups,
    group_reg=5,
    l1_reg=0,
    frobenius_lipschitz=True,
    scale_reg="inverse_group_size",
    subsampling_scheme=1,
    supress_warning=True,
    n_iter=1000,
    tol=1e-3,
)
gl.fit(X, y)
# https://group-lasso.readthedocs.io/en/latest/auto_examples/example_logistic_group_lasso.html

