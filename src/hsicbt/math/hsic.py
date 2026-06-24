import numpy as np
import torch


def sigma_estimation(X, Y):
    """sigma from median distance"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1e-2:
        med = 1e-2
    return med


def distmat(X):
    """distance matrix"""
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def kernelmat(X, sigma):
    """kernel matrix baker"""
    m = int(X.size()[0])
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])
    Dxx = distmat(X)

    # NOTE: the .type(torch.FloatTensor) casts below intentionally move the kernel
    # matrices to the CPU; the HSIC computation is pinned to CPU and changing this
    # would alter the numerics that reproduce the paper's results.
    if sigma:
        variance = 2.0 * sigma * sigma * X.size()[1]
        Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X, X)
            Kx = torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor)
        except RuntimeError as err:
            raise RuntimeError(
                f"Unstable sigma {sx} with maximum/minimum input ({torch.max(X)},{torch.min(X)})"
            ) from err

    Kxc = torch.mm(Kx, H)

    return Kxc


def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp(-X / (2.0 * sigma * sigma))
    return torch.mean(X)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx = torch.exp(-Dxx / (2.0 * sigma * sigma))  # kernel matrices
        Ky = torch.exp(-Dyy / (2.0 * sigma * sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x, x)
        sy = sigma_estimation(y, y)
        sxy = sigma_estimation(x, y)
        Kx = torch.exp(-Dxx / (2.0 * sx * sx))
        Ky = torch.exp(-Dyy / (2.0 * sy * sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x, y]))
    Dxy = Dxy[: x.size()[0], x.size()[0] :]
    Kxy = torch.exp(-Dxy / (1.0 * sxy * sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2 * torch.mean(Kxy)

    return mmdval


def mmd_pxpy_pxy(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx = torch.exp(-Dxx / (2.0 * sigma * sigma))  # kernel matrices
        Ky = torch.exp(-Dyy / (2.0 * sigma * sigma))
    else:
        sx = sigma_estimation(x, x)
        sy = sigma_estimation(y, y)
        Kx = torch.exp(-Dxx / (2.0 * sx * sx))
        Ky = torch.exp(-Dyy / (2.0 * sy * sy))
    A = torch.mean(Kx * Ky)
    B = torch.mean(torch.mean(Kx, dim=0) * torch.mean(Ky, dim=0))
    C = torch.mean(Kx) * torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2 * B + C
    return mmd_pxpy_pxy_val


def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy


def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """ """
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy / (Px * Py)
    return thehsic


def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True):
    """ """
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma)

    epsilon = 1e-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = Kxc.mm(Kxc_i)
    Ry = Kyc.mm(Kyc_i)
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy
