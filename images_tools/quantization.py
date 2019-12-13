import numpy as np
import pytest


def quantize(arr, min_val, max_val, levels, dtype=np.int64):
    """Quantize an array of (-inf, inf) to [0, levels-1].

    Args:
        arr (ndarray): Input array.
        min_val (scalar): Minimum value to be clipped.
        max_val (scalar): Maximum value to be clipped.
        levels (int): Quantization levels.
        dtype (np.type): The type of the quantized array.

    Returns:
        tuple: Quantized array.
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(
            'levels must be a positive integer, but got {}'.format(levels))
    if min_val >= max_val:
        raise ValueError(
            'min_val ({}) must be smaller than max_val ({})'.format(
                min_val, max_val))

    arr = np.clip(arr, min_val, max_val) - min_val
    quantized_arr = np.minimum(
        np.floor(levels * arr / (max_val - min_val)).astype(dtype), levels - 1)

    return quantized_arr


def dequantize(arr, min_val, max_val, levels, dtype=np.float64):
    """Dequantize an array.

    Args:
        arr (ndarray): Input array.
        min_val (scalar): Minimum value to be clipped.
        max_val (scalar): Maximum value to be clipped.
        levels (int): Quantization levels.
        dtype (np.type): The type of the dequantized array.

    Returns:
        tuple: Dequantized array.
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(
            'levels must be a positive integer, but got {}'.format(levels))
    if min_val >= max_val:
        raise ValueError(
            'min_val ({}) must be smaller than max_val ({})'.format(
                min_val, max_val))

    dequantized_arr = (arr + 0.5).astype(dtype) * (
        max_val - min_val) / levels + min_val

    return dequantized_arr


def test_quantize():
    arr = np.random.randn(10, 10)
    levels = 20

    qarr = quantize(arr, -1, 1, levels)
    assert qarr.shape == arr.shape
    assert qarr.dtype == np.dtype('int64')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ref = min(levels - 1,
                      int(np.floor(10 * (1 + max(min(arr[i, j], 1), -1)))))
            assert qarr[i, j] == ref

    qarr = quantize(arr, -1, 1, 20, dtype=np.uint8)
    assert qarr.shape == arr.shape
    assert qarr.dtype == np.dtype('uint8')

    with pytest.raises(ValueError):
        quantize(arr, -1, 1, levels=0)
    with pytest.raises(ValueError):
        quantize(arr, -1, 1, levels=10.0)
    with pytest.raises(ValueError):
        quantize(arr, 2, 1, levels)


def test_dequantize():
    levels = 20
    qarr = np.random.randint(levels, size=(10, 10))

    arr = dequantize(qarr, -1, 1, levels)
    assert arr.shape == qarr.shape
    assert arr.dtype == np.dtype('float64')
    for i in range(qarr.shape[0]):
        for j in range(qarr.shape[1]):
            assert arr[i, j] == (qarr[i, j] + 0.5) / 10 - 1

    arr = dequantize(qarr, -1, 1, levels, dtype=np.float32)
    assert arr.shape == qarr.shape
    assert arr.dtype == np.dtype('float32')

    with pytest.raises(ValueError):
        dequantize(arr, -1, 1, levels=0)
    with pytest.raises(ValueError):
        dequantize(arr, -1, 1, levels=10.0)
    with pytest.raises(ValueError):
        dequantize(arr, 2, 1, levels)


def test_joint():
    arr = np.random.randn(100, 100)
    levels = 1000
    qarr = quantize(arr, -1, 1, levels)
    recover = dequantize(qarr, -1, 1, levels)
    assert np.abs(recover[arr < -1] + 0.999).max() < 1e-6
    assert np.abs(recover[arr > 1] - 0.999).max() < 1e-6
    assert np.abs((recover - arr)[(arr >= -1) & (arr <= 1)]).max() <= 1e-3

    arr = np.clip(np.random.randn(100) / 1000, -0.01, 0.01)
    levels = 99
    qarr = quantize(arr, -1, 1, levels)
    recover = dequantize(qarr, -1, 1, levels)
    assert np.all(recover == 0)