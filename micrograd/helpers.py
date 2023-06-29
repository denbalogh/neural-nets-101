def one_hot_encode(x, n_labels):
    return [1 if i == x else 0 for i in range(n_labels)]

def one_hot_decode(x):
    return x.index(max(x))

def one_hot_encode_labels(N):
    def _one_hot_encode_labels(example):
        example['label'] = one_hot_encode(example['label'], N)
        return example
    return _one_hot_encode_labels
