# import torch

# class Lattice:
#     def __init__(self,size,requires_grad=False):
#         self.size=size
#         self.requires_grad=requires_grad
#         #detaching the tensor from pytorch's internal computation graph
#         self.lattice=torch.zeros(size,requires_grad=requires_grad).detach()
#         self.grad = self.lattice.grad if requires_grad else None
#         self._ctx = None
#     def
