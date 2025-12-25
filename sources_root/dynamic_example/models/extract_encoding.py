import torch
from einops import rearrange

# model = DetectionPretrain(getCfg().config.train_model)


from collections import OrderedDict

from torch import nn


# Define a recursive function to register the hook on all modules
def register_hook_recursively(module, fn):
    if isinstance(module, torch.nn.Module):
        if fn is None:# Remove all forward hooks
            module._forward_hooks = OrderedDict()
        else:
            module.register_forward_hook(fn)
        for child in module.children():
            register_hook_recursively(child, fn)

class HookEmb(torch.nn.Module):
    def __init__(self, num_stages, out_dim, emb_dim, stride = [4, 2, 1], ksize = [5, 3, 1]):
        super().__init__()
        self.cur_emb = []
        self.reduce = torch.nn.ModuleList([torch.nn.LazyConv2d(out_dim - emb_dim, ksize[i], stride = stride[i], padding = (ksize[i] - stride[i] + 1) // 2) for i in range(num_stages)])
        self.emb = torch.nn.ModuleList([PositionalEmbedding2d(emb_dim) for _ in range(num_stages)])

    def clean(self):
        self.cur_emb = []
    def forward(self):
        out = []
        for i, c in enumerate(self.cur_emb):
            x = self.emb[i](self.reduce[i](c))
            x = rearrange(x, "b c h w -> b (h w) c")
            out.append(x)
        out = torch.cat(out, dim = 1)
        return out

    def append(self, x):
        self.cur_emb.append(x)

    def attach(self, model, cls_name): # "C3shakedrop"
        def forward_hook(module, _, out):
            if module.__class__.__name__ == cls_name:
                self.append(out)
                # print(f"Forward hook called for module: {module.__class__.__name__}, {output.shape if isinstance(output, torch.Tensor) else 'tuple'}, {input.shape if isinstance(input, torch.Tensor) else 'tuple'}")
        register_hook_recursively(model, forward_hook)



# class HookEmb(torch.nn.Module):

# Define a custom hook function

#
# register_hook_recursively(model.detector.model.detector, forward_hook)
# print(model.detector.model.detector(torch.randn((1,3,640, 640)))[0].shape)
class PositionalEmbedding2d(nn.Module):
    def __init__(self, embedding_dim):
        super(PositionalEmbedding2d, self).__init__()

        self.embedding_dim = embedding_dim
        self.row_embeddings = torch.nn.UninitializedParameter()
        self.col_embeddings = torch.nn.UninitializedParameter()

    def initialize_embeddings(self, image_height, image_width):
        self.row_embeddings = nn.Parameter(torch.zeros(image_height, 1, self.embedding_dim // 2))
        self.col_embeddings = nn.Parameter(torch.zeros(1, image_width, self.embedding_dim // 2))

        nn.init.xavier_uniform_(self.row_embeddings)
        nn.init.xavier_uniform_(self.col_embeddings)

    def forward(self, x):
        batch_size, _, image_height, image_width = x.size()

        if isinstance(self.row_embeddings, torch.nn.UninitializedParameter):
            self.initialize_embeddings(image_height, image_width)

        row_positions = torch.arange(image_height, device=x.device).float() / (image_height - 1)
        col_positions = torch.arange(image_width, device=x.device).float() / (image_width - 1)

        row_positions = 2 * row_positions - 1
        col_positions = 2 * col_positions - 1
        grid_x, grid_y = torch.meshgrid(row_positions, col_positions)
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1)

        positional_embeddings = torch.cat([self.row_embeddings + grid_x, self.col_embeddings + grid_y], dim=3)
        positional_embeddings = positional_embeddings.repeat(batch_size, 1, 1, 1)
        return torch.cat([x, positional_embeddings.permute(0, 3, 1, 2)], dim=1)
