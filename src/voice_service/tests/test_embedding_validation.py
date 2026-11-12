import numpy as np
import pytest

from voice_service.schemas import DIMENSION
from voice_service.store import VectorIndexedBase


def make_valid_embedding() -> list[float]:
    rng = np.random.default_rng(42)
    return rng.random(DIMENSION).astype(np.float32).tolist()


def test_valid_embedding():
    vector = make_valid_embedding()
    result = VectorIndexedBase._validate_embedding(vector)
    
    assert result.shape == (DIMENSION,)
    assert np.isfinite(result).all()


def test_zero_vector_rejected():
    zero_vector = [0.0] * DIMENSION
    
    with pytest.raises(ValueError, match="Zero embedding is not supported"):
        VectorIndexedBase._validate_embedding(zero_vector)


def test_nan_rejected():
    vector = make_valid_embedding()
    vector[0] = float("nan")
    
    with pytest.raises(ValueError, match="Embedding contains NaN or infinity"):
        VectorIndexedBase._validate_embedding(vector)


def test_infinity_rejected():
    vector = make_valid_embedding()
    vector[0] = float("inf")
    
    with pytest.raises(ValueError, match="Embedding contains NaN or infinity"):
        VectorIndexedBase._validate_embedding(vector)


def test_wrong_dimension_rejected():
    wrong_vector = [0.1] * 100
    
    with pytest.raises(ValueError, match=f"Embedding must have {DIMENSION} dimensions"):
        VectorIndexedBase._validate_embedding(wrong_vector)


def test_empty_list_rejected():
    with pytest.raises(ValueError):
        VectorIndexedBase._validate_embedding([])