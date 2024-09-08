import torch


class PyTorchBasics:
    """
    Implement the following python code with PyTorch.
    Use PyTorch functions to make your solution efficient and differentiable.

    General Rules:
    - No loops, no function calls (except for torch functions), no if statements
    - No numpy
    - PyTorch and tensor operations only
    - No assignments to results x[1] = 5; return x
    - A solution requires less than 10 PyTorch commands

    The grader will convert your solution to torchscript and make sure it does not
    use any unsupported operations (loops etc).
    """

    @staticmethod
    def make_it_pytorch_1(x: torch.Tensor) -> torch.Tensor:
        return x[::3]
       
    @staticmethod
    def make_it_pytorch_2(x: torch.Tensor) -> torch.Tensor: 
        values, indicies = torch.max(x,dim=2)
        return values

    @staticmethod
    def make_it_pytorch_3(x: torch.Tensor) -> torch.Tensor:
        return torch.unique(x,sorted= True)

    @staticmethod
    def make_it_pytorch_4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.where(y>torch.mean(x),1,0))

    @staticmethod
    def make_it_pytorch_5(x: torch.Tensor) -> torch.Tensor:
        return x.T

    @staticmethod
    def make_it_pytorch_6(x: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(x,0)

    @staticmethod
    def make_it_pytorch_7(x: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(torch.flip(x,dims=[1]))

    @staticmethod
    def make_it_pytorch_8(x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(x,0)

    @staticmethod
    def make_it_pytorch_9(x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(torch.cumsum(x, dim=0), dim=1)

    @staticmethod
    def make_it_pytorch_10(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.where(x < c , torch.tensor(0), x)

    @staticmethod
    def make_it_pytorch_11(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(x < c, as_tuple=False).T

    @staticmethod
    def make_it_pytorch_12(x: torch.Tensor, m: torch.BoolTensor) -> torch.Tensor:
        return torch.masked_select(x, m)

    @staticmethod
    def make_it_pytorch_extra_1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.diff(torch.cat((x,y)))

    @staticmethod
    def make_it_pytorch_extra_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.tensor((torch.abs(x.unsqueeze(1) - y.unsqueeze(0)) < 1e-3).any(dim=1).sum())
