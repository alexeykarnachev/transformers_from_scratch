import torch


def apply_attention_distr(
        values: torch.Tensor,
        distr: torch.Tensor
) -> torch.Tensor:
    # values: (bs, seq_len, n_heads, head_dim)
    # distr: (bs, n_heads, seq_len, seq_len)

    # Weighted multi-head values (bs, n_heads, head_dim, seq_len):
    v = distr @ values

    # Restore original shape (bs, seq_len, dim):
    v = v.transpose(1, 2).contiguous()
    v = v.view(*v.size()[:-2], -1)

    return v

# def get_positional_encoding_weights(dim: int, n_pos: int) -> torch.Tensor:
#     pos_vec = torch.arange(start=0, end=n_pos)
#     pos_mat = pos_vec.unsqueeze(-1).repeat(1, dim)
#
#     dim_vec = torch.arange(start=0, end=dim)
#     dim_mat = dim_vec.unsqueeze(0).repeat(n_pos, 1)
#
#     weights = torch.zeros(n_pos, dim)
#
#
#     powers = torch.arange(0, dim)
#
#     weights[:, ::2] = torch.sin(pos_vec[::2,:] / (10000 ** ))
#
#     print(dim_mat)
#     print(dim_mat.size())
#
# get_positional_encoding_weights(30, 10)