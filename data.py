import torch


def addition_mod_p_data(p: int, eq_token: int, op_token: int):
    x = torch.arange(0, p)
    y = torch.arange(0, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    x, y, labels = (lambda x, y, p: (x, y, (x + y) % p))(x, y, p)
    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels


def k_addition_mod_p_data(p: int, K: int, eq_token: int, op_token: int):
    x0 = torch.arange(0, p)
    x = torch.cartesian_prod(*[x0] * K).T

    eq = torch.ones_like(x[1]) * eq_token
    op = torch.ones_like(x[1]) * op_token

    labels = torch.sum(x, dim=0) % p
    inputs = torch.cat([x, op.unsqueeze(0), eq.unsqueeze(0)], dim=0)

    return inputs.T, labels


def cycle(dl):
    while True:
        for data in dl:
            yield data
