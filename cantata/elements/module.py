import torch


class Module(torch.nn.Module):
    '''
    Module base
    '''

    def __init__(self):
        super().__init__()

    def copy_from(self, other):
        for pname, oldval in self.named_parameters():
            assert False, f'Cannot call copy_from on a model with Parameters,\
 but found {pname}.'

        for bname, oldval in self.named_buffers(recurse=False):
            setattr(self, bname, getattr(other, bname).clone())

        for cname, child in self.named_children():
            child.copy_from(getattr(other, cname))