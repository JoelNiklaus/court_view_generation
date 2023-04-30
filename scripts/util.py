def get_batch_size(model_type, gpu_memory, input_length, output_length):
    batch_sizes_same = {
        'mgpt': {
            24: {512: 0, 1024: 0, 2048: 0},
            48: {512: 1, 1024: 1, 2048: 1},  # never tested
            80: {512: 8, 1024: 6, 2048: 2},  # tested 1024 and 2048
        },
        'mt5-small': {
            24: {512: 4, 1024: 2, 2048: 1},  # tested with 512, 1024 and 2048 seq length
            48: {512: 8, 1024: 4, 2048: 2},  # never tested
            80: {512: 16, 1024: 12, 2048: 4},  # tested 1024 and 2048
        },
        'mt5-base': {
            24: {512: 2, 1024: 1, 2048: 1},  # only tested with 512 seq length
            48: {512: 4, 1024: 2, 2048: 1},  # never tested
            80: {512: 8, 1024: 6, 2048: 2},  # tested 1024 and 2048
        },
        'mt5-large': {
            24: {512: 0, 1024: 0, 2048: 0},  # tested
            48: {512: 1, 1024: 1, 2048: 1},  # never tested
            80: {512: 4, 1024: 2, 2048: 1},  # tested 1024 and 2048
        },
    }

    batch_sizes_512_output = {
        'mt5-small': {
            24: {2048: 1, 3072: 1, 4096: 1},  # never tested
            48: {2048: 2, 3072: 1, 4096: 1},  # never tested
            80: {2048: 14, 3072: 10, 4096: 6},  # tested
        },
        'mt5-base': {
            24: {2048: 1, 3072: 1, 4096: 1},  # never tested
            48: {2048: 1, 3072: 1, 4096: 0},  # never tested
            80: {2048: 6, 3072: 4, 4096: 2},  # tested
        },
        'mt5-large': {
            24: {2048: 0, 3072: 0, 4096: 0},  # never tested
            48: {2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {2048: 2, 3072: 1, 4096: 0},  # tested
        },
    }

    try:
        if input_length == output_length:
            batch_sizes = batch_sizes_same
        elif output_length == 512:
            batch_sizes = batch_sizes_512_output
        else:
            raise ValueError(f"Output length {output_length} not supported")
        batch_size = batch_sizes[model_type][gpu_memory][input_length]
    except KeyError:
        print(f"Batch size not found for model type: {model_type}, input length: {input_length}, gpu memory: {gpu_memory}")
        raise KeyError

    return batch_size
