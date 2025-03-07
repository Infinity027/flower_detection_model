
import torch

def set_grid(grid_size=7):
    return torch.meshgrid((torch.arange(grid_size), torch.arange(grid_size)), indexing="ij")

if __name__ == "__main__":
    grid_x, grid_y = set_grid()

    print(grid_x.contiguous().view((1, -1)))
