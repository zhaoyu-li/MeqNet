import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
import timeit
# import mixnet._cpp

if torch.cuda.is_available(): import mixnet._cuda


def get_k(n):
    return int((2 * n) ** 0.5 + 1)


class MixingFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, C, z, is_input, max_iter, eps, prox_lam):
        B, n = z.size(0), C.size(0)
        k = 32 # int((2 * n) ** 0.5 + 3) // (4*4)
        ctx.prox_lam = prox_lam

        assert(C.is_cuda)
        device = 'cuda'

        ctx.g, ctx.gnrm = torch.zeros(B,k, device=device), torch.zeros(B,n, device=device)
        ctx.index = torch.zeros(B,n, dtype=torch.int, device=device)
        ctx.is_input = torch.zeros(B,n, dtype=torch.int, device=device)
        ctx.V, ctx.W = torch.zeros(B,n,k, device=device).normal_(), torch.zeros(B,k,n, device=device)
        ctx.z = torch.zeros(B,n, device=device)
        ctx.niter = torch.zeros(B, dtype=torch.int, device=device)

        ctx.C = torch.zeros(n,n, device=device)
        ctx.Cdiags = torch.zeros(n, device=device)

        ctx.z[:] = z.data
        ctx.C[:] = C.data
        ctx.is_input[:] = is_input.data

        perm = torch.randperm(n-1, dtype=torch.int, device=device)

        satnet_impl = mixnet._cuda
        satnet_impl.init(perm, ctx.is_input, ctx.index, ctx.z, ctx.V)

        ctx.W[:] = ctx.V.transpose(1, 2)
        ctx.Cdiags[:] = torch.diagonal(C)

        satnet_impl.forward(max_iter, eps, 
                ctx.index, ctx.niter, ctx.C, ctx.z, 
                ctx.V, ctx.W, ctx.gnrm, ctx.Cdiags, ctx.g)

        return ctx.z.clone()
    
    @staticmethod
    def backward(ctx, dz):
        B, n = dz.size(0), ctx.C.size(0)
        k = 32 # int((2 * n) ** 0.5 + 3) // (4*4)

        assert(ctx.C.is_cuda)
        device = 'cuda'

        ctx.dC = torch.zeros(B,n,n, device=device)
        ctx.U, ctx.Phi = torch.zeros(B,n,k, device=device), torch.zeros(B,k,n, device=device)
        ctx.dz = torch.zeros(B,n, device=device)

        ctx.dz[:] = dz.data

        satnet_impl = mixnet._cuda
        satnet_impl.backward(ctx.prox_lam, 
                ctx.is_input, ctx.index, ctx.niter, ctx.C, ctx.dC, ctx.z, ctx.dz,
                ctx.V, ctx.U, ctx.W, ctx.Phi, ctx.gnrm, ctx.Cdiags, ctx.g)

        ctx.dC = ctx.dC.sum(dim = 0)

        return ctx.dC, ctx.dz, None, None, None, None


def insert_constants(x, pre, n_pre, app, n_app):
    ''' prepend and append torch tensors
    '''
    one = x.new(x.size()[0], 1).fill_(1)
    seq = []
    if n_pre != 0:
        seq.append((pre * one).expand(-1, n_pre))
    seq.append(x)
    if n_app != 0:
        seq.append((app * one).expand(-1, n_app))
    r = torch.cat(seq, dim=1)
    r.requires_grad = False
    return r


class MixNet(nn.Module):
    '''Apply a MixNet layer to complete the input probabilities.

    Args:
        n: Number of input variables.
        aux: Number of auxiliary variables.

        max_iter: Maximum number of iterations for solving
            the inner optimization problem.
            Default: 40
        eps: The stopping threshold for the inner optimizaiton problem.
            The inner Mixing method will stop when the function decrease
            is less then eps times the initial function decrease.
            Default: 1e-4
        prox_lam: The diagonal increment in the backward linear system
            to make the backward pass more stable.
            Default: 1e-2
        weight_normalize: Set true to perform normlization for init weights.
            Default: True

    Inputs: (z, is_input)
        **z** of shape `(batch, n)`:
            Float tensor containing the probabilities (must be in [0,1]).
        **is_input** of shape `(batch, n)`:
            Int tensor indicating which **z** is a input.

    Outputs: z
        **z** of shape `(batch, n)`:
            The prediction probabiolities.

    Attributes: C
        **S** of shape `(n, n)`:
            The learnable equality matrix containing `n` variables.

    Examples:
        >>> mix = mixnet.MixNet(3, aux=5)
        >>> z = torch.randn(2, 3)
        >>> is_input = torch.IntTensor([[1, 1, 0], [1,0,1]])
        >>> pred = mix(z, is_input)
    '''

    def __init__(self, n, aux=0, max_iter=40, eps=1e-4, prox_lam=1e-2, weight_normalize=True):
        super(MixNet, self).__init__()
        self.nvars = n + 1 + aux
        C_t = torch.randn(self.nvars, self.nvars)
        C_t = C_t + C_t.t() - 1
        C_t.fill_diagonal_(0)

        if weight_normalize: C_t = C_t * ((.5 / (self.nvars * 2)) ** 0.5) # extremely important!
        self.C = nn.Parameter(C_t)
        self.aux = aux
        self.max_iter, self.eps, self.prox_lam = max_iter, eps, prox_lam

    def forward(self, z, is_input):
        device = 'cuda' if self.C.is_cuda else 'cpu'
        is_input = insert_constants(is_input.data, 1, 1, 0, self.aux)
        z = torch.cat([torch.ones(z.size(0), 1, device=device), z, torch.zeros(z.size(0), self.aux, device=device)],
                      dim=1)

        z = MixingFunc.apply(self.C, z, is_input, self.max_iter, self.eps, self.prox_lam)

        return z[:, 1:self.C.size(0) - self.aux]

if __name__ == '__main__':
    C_t = torch.tensor(
        [[  0,   -2.58, -3.47, 1.61, -0.21, 6.84, ],
        [-2.58,   0,   2.82, -1.18, 0.29, -5.45, ],
        [-3.47, 2.82,   0,   -1.58, 0.70, -6.99, ],
        [1.61, -1.18, -1.58,   0,   -0.48, 2.62, ],
        [-0.21, 0.29, 0.70, -0.48,   0,   0.11, ],
        [6.84, -5.45, -6.99, 2.62, 0.11,   0,   ],]
    )
    mix = MixNet(3, aux=2)
    mix.C = nn.Parameter(C_t)
    z = torch.tensor([[0, 0, -1], [0, 1, -1], [1, 0, -1], [1, 1, -1]])
    is_input = torch.IntTensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    # label = torch.FloatTensor([[1, -1, -1, 1]])
    pred = mix(z, is_input)
    print(pred)
    # loss = nn.functional.mse_loss(pred, label)
    # loss = pred[0,2]+pred[0,3]
    # loss.backward()
    # print(mix.C.grad)
    # print(z.grad)
