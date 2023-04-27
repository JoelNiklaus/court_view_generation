def get_batch_size(model_type, gpu_memory, seq_length):
    batch_sizes = {
        'mt5-small': {
            24: {512: 4, 1024: 2, 2048: 1, 3072: 1, 4096: 1},  # tested with 512, 1024 and 2048 seq length
            48: {512: 8, 1024: 4, 2048: 2, 3072: 1, 4096: 1},  # never tested
            80: {512: 16, 1024: 8, 2048: 4, 3072: 2, 4096: 1}, # never tested
            },
        'mt5-base': {
            24: {512: 2, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # only tested with 512 seq length
            48: {512: 4, 1024: 2, 2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {512: 8, 1024: 4, 2048: 2, 3072: 1, 4096: 1},  # never tested
            },
        'mt5-large': {
            24: {512: 0, 1024: 0, 2048: 0, 3072: 0, 4096: 0},  # tested
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {512: 2, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            },
        'mt5-xl': {
            24: {512: 0, 1024: 0, 2048: 0, 3072: 0, 4096: 0},
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {512: 2, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            },
        'mt5-xxl': {
            24: {512: 0, 1024: 0, 2048: 0, 3072: 0, 4096: 0},
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            },
        'mgpt': {
            24: {512: 0, 1024: 0, 2048: 0, 3072: 0, 4096: 0},
            48: {512: 1, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            80: {512: 2, 1024: 1, 2048: 1, 3072: 1, 4096: 1},  # never tested
            }
        }

    try:
        batch_size = batch_sizes[model_type][gpu_memory][seq_length]
    except KeyError:
        print(f"Batch size not found for model type: {model_type}, seq length: {seq_length}, gpu memory: {gpu_memory}")
        raise KeyError

    return batch_size