import numpy
import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


def _test_max_grad(t, out, dim):
    t.zero_grad_()
    out.sum().backward()
    for ind in out._tensor.indices():
        for i in range(t.shape[dim]):
            now_ind = list(ind)
            now_ind[dim] = i
            now_ind = tuple(now_ind)
            if t[now_ind] == out[ind]:
                assert_close(t.grad[now_ind], 1.0)
            else:
                assert_close(t.grad[now_ind], 0.0)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    t.requires_grad_(True)
    out = minitorch.max(t, 0)
    for ind in out._tensor.indices():
        exp_out_ele = max([t[i, ind[1], ind[2]] for i in range(t.shape[0])])
        assert_close(out[ind], exp_out_ele)
    _test_max_grad(t, out, 0)

    out = minitorch.max(t, 1)
    for ind in out._tensor.indices():
        exp_out_ele = max([t[ind[0], i, ind[2]] for i in range(t.shape[1])])
        assert_close(out[ind], exp_out_ele)
    _test_max_grad(t, out, 1)

    out = minitorch.max(t, 2)
    for ind in out._tensor.indices():
        exp_out_ele = max([t[ind[0], ind[1], i] for i in range(t.shape[2])])
        assert_close(out[ind], exp_out_ele)
    _test_max_grad(t, out, 2)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    import random

    random.seed(1)
    q = minitorch.dropout(t, 0.5, ignore=False)
    cnt_t = numpy.count_nonzero(t._tensor._storage == 0.0)
    cnt_q = numpy.count_nonzero(q._tensor._storage == 0.0)
    assert cnt_q >= cnt_t


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
