"""pytest 公共夹具。

已发布定义集带磁盘持久化（app/unit_sets.py），测试中把进程级存储单例
重定向到每个用例独立的临时目录，既避免污染开发机上的默认数据目录，
也保证用例之间互不干扰。
"""

from __future__ import annotations

import pytest

from app import unit_sets


@pytest.fixture(autouse=True)
def _isolated_unit_set_store(tmp_path):
    store = unit_sets.UnitSetStore(str(tmp_path / "data"))
    store.load()
    unit_sets._store = store
    yield store
    unit_sets._store = None
