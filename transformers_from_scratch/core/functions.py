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
