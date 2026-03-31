import argparse

import torch


def parse_seq_lens(seq_lens_arg: str, num_requests: int) -> list[int]:
    seq_lens = [int(x.strip()) for x in seq_lens_arg.split(",") if x.strip()]
    if len(seq_lens) != num_requests:
        raise ValueError(
            f"--seq-lens must provide exactly {num_requests} entries, got {len(seq_lens)}"
        )
    if any(x <= 0 for x in seq_lens):
        raise ValueError("--seq-lens values must all be positive")
    return seq_lens


def build_runtime_tensors(
    seq_lens: list[int],
    max_seq_length: int,
    max_num_batched_tokens: int,
    max_num_pages: int,
    page_size: int,
):
    num_requests = len(seq_lens)
    max_prompt_len = max(seq_lens)
    if max_seq_length < max_prompt_len + 1:
        raise ValueError(
            f"max_seq_length={max_seq_length} is too small for max cached length "
            f"{max_prompt_len} plus one decode token"
        )
    if max_num_batched_tokens < num_requests:
        raise ValueError(
            f"max_num_batched_tokens={max_num_batched_tokens} must be at least "
            f"the number of requests={num_requests} for one-token decode"
        )

    pages_per_request = [(seq_len + 1 + page_size - 1) // page_size for seq_len in seq_lens]
    total_pages = sum(pages_per_request)
    if total_pages > max_num_pages:
        raise ValueError(
            f"Need {total_pages} KV pages for seq_lens={seq_lens}, but max_num_pages="
            f"{max_num_pages}"
        )

    tokens = torch.zeros((num_requests, max_seq_length), dtype=torch.long, device="cuda")
    prompt_lengths = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    step = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.ones((num_requests,), dtype=torch.int32, device="cuda")
    input_tokens = torch.zeros((max_num_batched_tokens, 1), dtype=torch.long, device="cuda")
    output_tokens = torch.zeros((max_num_batched_tokens, 1), dtype=torch.long, device="cuda")

    for request_id, seq_len in enumerate(seq_lens):
        for pos in range(seq_len):
            tokens[request_id, pos] = (request_id * 97 + pos) % 32000
        tokens[request_id, seq_len] = (request_id * 97 + seq_len) % 32000

    qo_indptr = torch.arange(
        0, num_requests + 1, dtype=torch.int32, device="cuda"
    )
    paged_kv_indptr = torch.zeros((num_requests + 1,), dtype=torch.int32, device="cuda")
    running_pages = 0
    for idx, num_pages in enumerate(pages_per_request):
        paged_kv_indptr[idx] = running_pages
        running_pages += num_pages
    paged_kv_indptr[num_requests] = running_pages

    paged_kv_indices = torch.arange(
        max_num_pages, dtype=torch.int32, device="cuda"
    )
    paged_kv_last_page_len = torch.tensor(
        [((seq_len + 1 - 1) % page_size) + 1 for seq_len in seq_lens],
        dtype=torch.int32,
        device="cuda",
    )

    return {
        "tokens": tokens,
        "prompt_lengths": prompt_lengths,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "step": step,
        "num_new_tokens": num_new_tokens,
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_indices_buffer": paged_kv_indices,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trace-name", default="synthetic_attention")
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--max-num-batched-requests", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--max-num-pages", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-q-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--split-kv", action="store_true")
    parser.add_argument(
        "--seq-lens",
        default="8,16,24,32,40,48,56,64",
        help="Comma-separated cached KV length for each request",
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    args = parser.parse_args()
    args.max_num_pages = 64
    args.max_num_batched_requests = 32
    args.max_num_batched_tokens = 32
    args.max_seq_length = 512

    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    # seq_lens = parse_seq_lens(args.seq_lens, args.max_num_batched_requests)
    seq_lens = [511] * 32

    try:
        import mirage as mi
    except ImportError as exc:
        raise RuntimeError("Failed to import mirage") from exc

    rank = 0
    world_size = 1
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)

    num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
    print("Input arguments:", args)
    print(f"Per-request cached lengths: {seq_lens}")
    print(f"num_workers={num_workers} num_schedulers={num_schedulers}")

    runtime_tensors = build_runtime_tensors(
        seq_lens=seq_lens,
        max_seq_length=args.max_seq_length,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
    )

    if args.profiling:
        profiler_tensor = torch.zeros(
            3000 * 128, dtype=torch.uint64, device="cuda"
        ).contiguous()
    else:
        profiler_tensor = None

    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=world_size,
        mpi_rank=rank,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        eos_token_id=0,
        meta_tensors=runtime_tensors,
        profiler_tensor=profiler_tensor,
        trace_name=args.trace_name,
        spec_decode_config=None,
        use_cutlass_kernel=False,
    )

    fused_qkv_dim = (args.num_q_heads + 2 * args.num_kv_heads) * args.head_dim
    num_kv_cache_chunks = max(1, args.max_seq_length // 256)
    attn_input_torch = torch.randn(
        (args.max_num_batched_tokens, fused_qkv_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_norm_torch = torch.randn((args.head_dim,), dtype=torch.bfloat16, device="cuda")
    k_norm_torch = torch.randn((args.head_dim,), dtype=torch.bfloat16, device="cuda")
    k_cache_torch = torch.randn(
        (args.max_num_pages, args.page_size, args.num_kv_heads, args.head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache_torch = torch.randn(
        (args.max_num_pages, args.page_size, args.num_kv_heads, args.head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    cos_pos_embed_torch = torch.randn(
        (args.max_seq_length, args.head_dim), dtype=torch.bfloat16, device="cuda"
    )
    sin_pos_embed_torch = torch.randn(
        (args.max_seq_length, args.head_dim), dtype=torch.bfloat16, device="cuda"
    )
    attn_out_torch = torch.zeros(
        (args.max_num_batched_tokens, args.num_q_heads * args.head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )

    attn_input = mpk.attach_input(torch_tensor=attn_input_torch, name="attn_in")
    q_norm = mpk.attach_input(torch_tensor=q_norm_torch, name="layer_0_q_norm")
    k_norm = mpk.attach_input(torch_tensor=k_norm_torch, name="layer_0_k_norm")
    k_cache = mpk.attach_input(torch_tensor=k_cache_torch, name="layer_0_k_cache")
    v_cache = mpk.attach_input(torch_tensor=v_cache_torch, name="layer_0_v_cache")
    cos_pos_embed = mpk.attach_input(
        torch_tensor=cos_pos_embed_torch, name="cos_position_embedding"
    )
    sin_pos_embed = mpk.attach_input(
        torch_tensor=sin_pos_embed_torch, name="sin_position_embedding"
    )
    attn_out = mpk.attach_input(torch_tensor=attn_out_torch, name="attn_out")

    if args.split_kv:
        lse = mpk.new_tensor(
            dims=(
                args.max_num_batched_tokens,
                num_kv_cache_chunks * args.num_q_heads // args.num_kv_heads,
                args.num_kv_heads,
            ),
            strides=(
                num_kv_cache_chunks * args.num_q_heads,
                1,
                num_kv_cache_chunks * args.num_q_heads // args.num_kv_heads,
            ),
            dtype=mi.float32,
            name="lse",
            io_category="cuda_tensor",
        )
        attn_out_tmp = mpk.new_tensor(
            dims=(
                args.max_num_batched_tokens,
                num_kv_cache_chunks * args.num_q_heads // args.num_kv_heads * args.head_dim,
                args.num_kv_heads,
            ),
            strides=(
                num_kv_cache_chunks * args.num_q_heads,
                1,
                num_kv_cache_chunks * args.num_q_heads // args.num_kv_heads * args.head_dim,
            ),
            dtype=mi.bfloat16,
            name="attn_out_tmp",
            io_category="cuda_tensor",
        )

        mpk.paged_attention_split_kv_layer(
            input=attn_input,
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_pos_embed=cos_pos_embed,
            sin_pos_embed=sin_pos_embed,
            lse=lse,
            output=attn_out_tmp,
            attention_params=(args.num_q_heads, num_kv_cache_chunks),
            grid_dim=(
                args.max_num_batched_requests,
                args.num_kv_heads,
                num_kv_cache_chunks,
            ),
            block_dim=(128, 1, 1),
        )
        mpk.paged_attention_split_kv_merge_layer(
            lse=lse,
            output_tmp=attn_out_tmp,
            output=attn_out,
            attention_params=(args.num_q_heads, num_kv_cache_chunks),
            grid_dim=(args.max_num_batched_requests, args.num_kv_heads, 1),
            block_dim=(128, 1, 1),
        )
    else:
        mpk.paged_attention_layer(
            input=attn_input,
            k_cache=k_cache,
            v_cache=v_cache,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_pos_embed=cos_pos_embed,
            sin_pos_embed=sin_pos_embed,
            output=attn_out,
            grid_dim=(args.max_num_batched_requests, args.num_kv_heads, 1),
            block_dim=(128, 1, 1),
        )

    results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
    with open(f"task_graph_{rank}.json", "w") as f:
        f.write(results["json_file"])
    with open(f"kernel_{rank}.cu", "w") as f:
        f.write(results["cuda_code"])

    mpk.compile(output_dir=args.output_dir)

    for _ in range(args.warmup_iters):
        mpk()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    for _ in range(args.benchmark_iters):
        runtime_tensors["step"].copy_(runtime_tensors["prompt_lengths"])
        runtime_tensors["num_new_tokens"].fill_(1)
        mpk()
    ender.record()
    torch.cuda.synchronize()

    total_ms = starter.elapsed_time(ender)
    avg_ms = total_ms / args.benchmark_iters
    print(f"Average synthetic attention decode latency: {avg_ms:.6f} ms")
    print(
        "Per-step shape: "
        f"requests={args.max_num_batched_requests}, q_heads={args.num_q_heads}, "
        f"kv_heads={args.num_kv_heads}, head_dim={args.head_dim}, split_kv={args.split_kv}"
    )


if __name__ == "__main__":
    main()
