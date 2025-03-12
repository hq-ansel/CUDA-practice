import triton.language as tl
import triton


@triton.jit
def element_mul(
    x_ptr,
    x_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    实际上是反向传播操作
    """
    # panel
    pid = tl.program_id(0)
    x_ptr += pid * x_stride
    grad_output = tl.load(grad_output_ptr)
    for i in range(0,n_cols,BLOCK_SIZE):
        x_offsets = i + tl.arange(0,BLOCK_SIZE)
        x_block = tl.load(x_ptr+x_offsets,mask=x_offsets<n_cols)
        tl.store(x_ptr+x_offsets, x_block*grad_output, mask=x_offsets<n_cols)
        