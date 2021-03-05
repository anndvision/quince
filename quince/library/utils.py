def lambda_top_func(mu, k, y, alpha):
    m = y.shape[0]
    r = (y[k:] - mu).sum(dim=0)
    py = (k + 1) / m
    return mu + r.div(m * (alpha + 1) - py)


def lambda_bottom_func(mu, k, y, alpha):
    m = y.shape[0]
    r = (y[: k + 1] - mu).sum(dim=0)
    py = k + 1
    return mu + r.div(m * alpha + py)


def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_ ** -1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_


def policy_risk(pi, y1, y0):
    return (pi * y1 + (1 - pi) * y0).mean()
