import numpy as np


def transpose_list_tuple(data: list[tuple]):
    if not data:
        return tuple()

    num_fields = len(data[0])
    for d in data:
        assert len(d) == num_fields, "すべてのタプルは同じフィールド数である必要があります"

    transposed = tuple(np.stack([d[i] for d in data], axis=0).astype(np.float32) for i in range(num_fields))
    return transposed


def main():
    data = [
        (np.array([67.0], dtype=np.float32), np.array([8.0, 5.0], dtype=np.float32)),
        (np.array([67.0], dtype=np.float32), np.array([8.0, 5.0], dtype=np.float32)),
        (np.array([77.0], dtype=np.float32), np.array([9.0, 5.0], dtype=np.float32)),
        (np.array([77.0], dtype=np.float32), np.array([9.0, 5.0], dtype=np.float32)),
        (np.array([67.0], dtype=np.float32), np.array([8.0, 5.0], dtype=np.float32)),
    ]

    result = transpose_list_tuple(data)
    print(result)


if __name__ == "__main__":
    main()
