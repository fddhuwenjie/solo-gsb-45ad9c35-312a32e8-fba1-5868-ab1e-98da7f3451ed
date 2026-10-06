"""可修订工作单（POST/GET/PATCH /revisable-worksheets）端到端测试。"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from app import revisions
from app.main import app

client = TestClient(app, raise_server_exceptions=True)

FOOT = {"symbol": "ft", "factor": "0.3048", "base_unit": "m"}


def var(value, unit):
    return {"value": value, "unit": unit}


def ev_step(name, formula, output_unit):
    return {"name": name, "formula": formula, "output_unit": output_unit}


def solve_step(name, equation, solve_for, output_unit):
    return {
        "name": name,
        "equation": equation,
        "solve_for": solve_for,
        "output_unit": output_unit,
    }


def publish(*items):
    resp = client.post("/custom-unit-sets", json={"custom_units": list(items)})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def make_body(steps, variables=None, **ref):
    body: dict = {"steps": steps}
    if variables is not None:
        body["variables"] = variables
    body.update(ref)
    return body


def create(
    steps,
    variables=None,
    *,
    set_id=None,
    alias=None,
):
    ref: dict = {}
    if set_id is not None:
        ref["custom_unit_set_id"] = set_id
    if alias is not None:
        ref["custom_unit_set_alias"] = alias
    body = make_body(steps, variables, **ref)
    return client.post("/revisable-worksheets", json=body)


def base_worksheet(**ref):
    # v*t = 120 km；s/2 = 60 km —— 两步，便于观察“下游不沿用旧结果”
    variables = {"v": var("60", "km/h"), "t": var("2", "h")}
    steps = [
        ev_step("s", "v*t", "km"),
        ev_step("half", "s/2", "km"),
    ]
    return steps, variables


def create_simple(set_id=None, alias=None):
    steps, variables = base_worksheet()
    kwargs = {"set_id": set_id if set_id is not None else publish(FOOT)}
    if alias is not None:
        kwargs = {"alias": alias}
    resp = create(steps, variables, **kwargs)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    return data["id"], data


def patch_var(worksheet_id, expected, name, value, unit):
    return client.patch(
        f"/revisable-worksheets/{worksheet_id}",
        json={"expected_revision": expected, "variable": {"name": name, "value": value, "unit": unit}},
    )


def patch_step(worksheet_id, expected, name, step):
    return client.patch(
        f"/revisable-worksheets/{worksheet_id}",
        json={"expected_revision": expected, "step": {"name": name, "step": step}},
    )


# ---------------------------------------------------------------------------
# 创建：整单成功才保存，固定定义集标识，初始修订号为 1
# ---------------------------------------------------------------------------


def test_create_with_set_id_persists_inputs_results_and_revision_one():
    set_id = publish(FOOT)
    steps, variables = base_worksheet()
    resp = create(steps, variables, set_id=set_id)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["id"].startswith("rws_")
    assert data["revision"] == 1
    assert data["custom_unit_set_id"] == set_id
    assert data["variables"] == {
        "t": {"value": "2", "unit": "h"},
        "v": {"value": "60", "unit": "km/h"},
    }
    assert [s["name"] for s in data["steps"]] == ["s", "half"]
    assert data["results"] == [
        {"name": "s", "value": "120", "unit": "km", "dimension": "m"},
        {"name": "half", "value": "60", "unit": "km", "dimension": "m"},
    ]


def test_create_returns_distinct_unpredictable_ids():
    ids = set()
    for _ in range(3):
        wid, _ = create_simple()
        assert wid.startswith("rws_")
        ids.add(wid)
    assert len(ids) == 3


def test_create_compute_failure_is_422_and_nothing_saved():
    steps, variables = base_worksheet()
    # 第二步量纲不兼容（s 是长度，加上时间）-> 整单失败
    steps[1] = ev_step("half", "s+t", "km")
    store = revisions.get_worksheet_store()
    before = store.count()
    resp = create(steps, variables, set_id=publish(FOOT))
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "dimension_mismatch"
    assert "results" not in resp.json()
    assert store.count() == before


def test_create_unknown_set_id_is_422():
    steps, variables = base_worksheet()
    resp = create(steps, variables, set_id="cus_does_not_exist")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"


def test_create_unknown_alias_is_422():
    steps, variables = base_worksheet()
    resp = create(steps, variables, alias="missing")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"


def test_create_without_either_set_reference_is_422():
    steps, variables = base_worksheet()
    resp = client.post(
        "/revisable-worksheets", json=make_body(steps, variables)
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"


def test_create_with_both_id_and_alias_is_422():
    set_id = publish(FOOT)
    steps, variables = base_worksheet()
    body = make_body(
        steps, variables,
        custom_unit_set_id=set_id, custom_unit_set_alias="lens",
    )
    resp = client.post("/revisable-worksheets", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"


def test_create_rejects_request_level_custom_units():
    steps, variables = base_worksheet()
    body = make_body(steps, variables, custom_units=[FOOT])
    resp = client.post("/revisable-worksheets", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"


def test_create_with_alias_pins_resolved_id():
    v1 = publish(FOOT)
    v2 = publish({"symbol": "ft", "factor": "0.304800", "base_unit": "m"})
    assert client.put(
        "/custom-unit-set-aliases/lens",
        json={"custom_unit_set_id": v1, "expected_id": None},
    ).status_code == 200
    resp = create(
        [
            ev_step("a", "x", "m"),
            ev_step("b", "a", "ft"),
        ],
        {"x": var("10", "ft")},
        alias="lens",
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["custom_unit_set_id"] == v1
    assert data["results"][0]["value"] == "3.048"


def test_worksheet_step_count_rules_apply_to_create():
    one = [ev_step("a", "1", "1")]
    resp = create(one, set_id=publish(FOOT))
    assert resp.status_code == 422
    six = [ev_step(f"s{i}", "1", "1") for i in range(6)]
    assert create(six, set_id=publish(FOOT)).status_code == 422


# ---------------------------------------------------------------------------
# 读取：按标识返回当前内容与修订号；不存在 404；快照一致
# ---------------------------------------------------------------------------


def test_get_returns_current_content():
    wid, created = create_simple()
    resp = client.get(f"/revisable-worksheets/{wid}")
    assert resp.status_code == 200
    assert resp.json() == created


def test_get_unknown_id_is_404():
    resp = client.get("/revisable-worksheets/rws_does_not_exist")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "worksheet_not_found"


def test_get_response_is_a_snapshot_unaffected_by_later_mutation():
    wid, before = create_simple()
    # 调用方拿到响应后再发起修订，旧响应对象（及读取值）仍描述修订 1
    assert patch_var(wid, 1, "t", "3", "h").status_code == 200
    after = client.get(f"/revisable-worksheets/{wid}").json()
    assert before["revision"] == 1
    assert after["revision"] == 2
    assert before["results"][0]["value"] == "120"
    assert after["results"][0]["value"] == "180"
    assert before["variables"]["t"] == {"value": "2", "unit": "h"}
    assert after["variables"]["t"] == {"value": "3", "unit": "h"}


# ---------------------------------------------------------------------------
# 修订：替换一个变量或一个步骤，整单重算、修订号递增、下游不沿用旧结果
# ---------------------------------------------------------------------------


def test_revise_variable_value_and_unit_reruns_whole_worksheet():
    wid, _ = create_simple()
    # t: 2 h -> 3 h：第一步 s 从 120 变为 180，下游 half 必须同步从 60 变为 90
    resp = patch_var(wid, 1, "t", "3", "h")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["revision"] == 2
    assert data["variables"]["t"] == {"value": "3", "unit": "h"}
    assert [r["value"] for r in data["results"]] == ["180", "90"]


def test_revise_variable_can_change_unit():
    # 120 min == 2 h：物理量相同，结果应保持 120/60
    wid, _ = create_simple()
    resp = patch_var(wid, 1, "t", "120", "min")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["revision"] == 2
    assert data["variables"]["t"] == {"value": "120", "unit": "min"}
    assert [r["value"] for r in data["results"]] == ["120", "60"]


def test_revise_step_formula_and_target_unit_keeps_name():
    wid, _ = create_simple()
    # 第二步改为 s/3 且输出到 m
    resp = patch_step(
        wid, 1, "half", {"name": "half", "formula": "s/3", "output_unit": "m"}
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["revision"] == 2
    assert data["steps"][1] == {"name": "half", "formula": "s/3", "output_unit": "m"}
    assert data["results"][1] == {"name": "half", "value": "40000", "unit": "m", "dimension": "m"}
    # 第一步结果不受影响
    assert data["results"][0]["value"] == "120"


def test_revise_step_can_switch_between_evaluate_and_solve():
    wid, _ = create_simple()
    # 把求值步骤 half 替换为同一步名的反求步骤：s/3 = x -> x=40 km
    resp = patch_step(
        wid,
        1,
        "half",
        solve_step("half", "s/3 = x", "x", "km"),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    step = data["steps"][1]
    assert step["name"] == "half"
    assert step["equation"] == "s/3 = x"
    assert step.get("formula") is None
    assert data["results"][1]["value"] == "40"


def test_multiple_revisions_increment_revision_and_rerun_each_time():
    wid, data = create_simple()
    assert data["results"][0]["value"] == "120"
    for expected, hours in ((1, "3"), (2, "4"), (3, "5")):
        resp = patch_var(wid, expected, "t", hours, "h")
        assert resp.status_code == 200, resp.text
        assert resp.json()["revision"] == expected + 1
        assert resp.json()["results"][0]["value"] == str(60 * int(hours))
        assert resp.json()["results"][1]["value"] == str(30 * int(hours))


def test_downstream_steps_never_reuse_stale_results():
    # 三步链路：s = v*t；half = s/2；quarter = half/2
    variables = {"v": var("60", "km/h"), "t": var("2", "h")}
    steps = [
        ev_step("s", "v*t", "km"),
        ev_step("half", "s/2", "km"),
        ev_step("quarter", "half/2", "km"),
    ]
    resp = create(steps, variables, set_id=publish(FOOT))
    assert resp.status_code == 201
    wid = resp.json()["id"]
    assert [r["value"] for r in resp.json()["results"]] == ["120", "60", "30"]
    resp = patch_var(wid, 1, "t", "4", "h")
    assert resp.status_code == 200
    # 三级下游全部基于新输入重算，而不是只改第一步
    assert [r["value"] for r in resp.json()["results"]] == ["240", "120", "60"]


# ---------------------------------------------------------------------------
# 修订的乐观并发：409 与 422 语义
# ---------------------------------------------------------------------------


def test_revise_with_stale_expected_revision_is_409_and_changes_nothing():
    wid, _ = create_simple()
    assert patch_var(wid, 1, "t", "3", "h").status_code == 200
    # 两个基于修订 1 的延迟修改：第一个已成功，第二个必须 409
    resp = patch_var(wid, 1, "t", "4", "h")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "revision_conflict"
    assert detail["current_revision"] == 2
    # 已保存内容仍是首个成功修订
    current = client.get(f"/revisable-worksheets/{wid}").json()
    assert current["revision"] == 2
    assert current["variables"]["t"]["value"] == "3"
    assert [r["value"] for r in current["results"]] == ["180", "90"]


def test_patch_unknown_id_is_404():
    resp = patch_var("rws_nope", 1, "t", "3", "h")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "worksheet_not_found"


def test_revise_must_target_existing_variable():
    wid, _ = create_simple()
    resp = patch_var(wid, 1, "missing", "1", "m")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"
    assert client.get(f"/revisable-worksheets/{wid}").json()["revision"] == 1


def test_revise_must_target_existing_step():
    wid, _ = create_simple()
    resp = patch_step(wid, 1, "nope", ev_step("nope", "1", "1"))
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"
    assert client.get(f"/revisable-worksheets/{wid}").json()["revision"] == 1


def test_revise_step_rename_rejected():
    wid, _ = create_simple()
    resp = patch_step(wid, 1, "half", ev_step("other", "1", "km"))
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"
    assert client.get(f"/revisable-worksheets/{wid}").json()["revision"] == 1


def test_revise_requires_exactly_one_change():
    wid, _ = create_simple()
    resp = client.patch(
        f"/revisable-worksheets/{wid}",
        json={"expected_revision": 1},
    )
    assert resp.status_code == 422
    resp = client.patch(
        f"/revisable-worksheets/{wid}",
        json={
            "expected_revision": 1,
            "variable": {"name": "t", "value": "3", "unit": "h"},
            "step": {"name": "half", "step": ev_step("half", "s", "km")},
        },
    )
    assert resp.status_code == 422
    assert client.get(f"/revisable-worksheets/{wid}").json()["revision"] == 1


def test_revise_extra_fields_forbidden():
    wid, _ = create_simple()
    resp = client.patch(
        f"/revisable-worksheets/{wid}",
        json={
            "expected_revision": 1,
            "variable": {"name": "t", "value": "3", "unit": "h"},
            "unexpected": 1,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_request"


def test_compute_failure_does_not_change_saved_revision_or_results():
    wid, before = create_simple()
    # t 改成长度量：第一步 v*t 量纲冲突 -> 422，修订号停留在 1
    resp = patch_var(wid, 1, "t", "1", "m")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "dimension_mismatch"
    after = client.get(f"/revisable-worksheets/{wid}").json()
    assert after["revision"] == 1
    assert after["variables"] == before["variables"]
    assert after["results"] == before["results"]
    assert after["steps"] == before["steps"]


def test_step_revision_compute_failure_leaves_everything_unchanged():
    wid, before = create_simple()
    # 第二步公式改成量纲冲突
    resp = patch_step(
        wid, 1, "half", {"name": "half", "formula": "s+t", "output_unit": "km"}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "dimension_mismatch"
    after = client.get(f"/revisable-worksheets/{wid}").json()
    assert after["revision"] == 1
    assert after["steps"] == before["steps"]
    assert after["results"] == before["results"]
    # 同一修订号仍可继续修订
    resp = patch_var(wid, 1, "t", "3", "h")
    assert resp.status_code == 200
    assert resp.json()["revision"] == 2


def test_precision_overflow_revision_is_422_without_revision_bump():
    # 先用可精确十进制的链路创建（a=1, b=2 -> r=0.5 -> big=1.5），
    # 再把 b 改为 3：第一步 r=1/3 无法精确十进制 -> 422，修订号不递增
    steps = [
        ev_step("r", "a/b", "1"),
        ev_step("big", "r*3", "1"),
    ]
    variables0 = {"a": var("1", "m"), "b": var("2", "m")}
    resp = create(steps, variables0, set_id=publish(FOOT))
    assert resp.status_code == 201, resp.text
    wid = resp.json()["id"]
    resp = patch_var(wid, 1, "b", "3", "m")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "precision_overflow"
    current = client.get(f"/revisable-worksheets/{wid}").json()
    assert current["revision"] == 1
    assert [r["value"] for r in current["results"]] == ["0.5", "1.5"]


# ---------------------------------------------------------------------------
# 定义集固定：别名切换后仍使用创建时的定义
# ---------------------------------------------------------------------------


def test_alias_switch_after_creation_does_not_affect_revisions():
    v1 = publish(FOOT)
    v2 = publish({"symbol": "ft", "factor": "0.304800", "base_unit": "m"})
    assert client.put(
        "/custom-unit-set-aliases/lens",
        json={"custom_unit_set_id": v1, "expected_id": None},
    ).status_code == 200
    resp = create(
        [ev_step("a", "x", "m"), ev_step("b", "a", "ft")],
        {"x": var("10", "ft")},
        alias="lens",
    )
    wid = resp.json()["id"]
    assert resp.json()["custom_unit_set_id"] == v1
    # 维护者切换别名到 v2（1 ft = 0.304800 m）
    assert client.put(
        "/custom-unit-set-aliases/lens",
        json={"custom_unit_set_id": v2, "expected_id": v1},
    ).status_code == 200
    # 工作单的读取与后续修订仍固定使用 v1：20 ft = 6.096 m（v1 口径）
    got = client.get(f"/revisable-worksheets/{wid}").json()
    assert got["custom_unit_set_id"] == v1
    resp = patch_var(wid, 1, "x", "20", "ft")
    assert resp.status_code == 200, resp.text
    assert resp.json()["custom_unit_set_id"] == v1
    assert resp.json()["results"][0]["value"] == "6.096"
    assert resp.json()["results"][1]["value"] == "20"


def test_definition_set_remains_usable_when_alias_deleted_by_repointing():
    # 别名指向别处后，工作单仍按固定标识正常计算与修订
    v1 = publish(FOOT)
    v2 = publish({"symbol": "zz", "factor": "2", "base_unit": "m"})
    assert client.put(
        "/custom-unit-set-aliases/ref",
        json={"custom_unit_set_id": v1, "expected_id": None},
    ).status_code == 200
    resp = create(
        [ev_step("a", "x", "m"), ev_step("b", "a", "ft")],
        {"x": var("1", "ft")},
        alias="ref",
    )
    wid = resp.json()["id"]
    assert client.put(
        "/custom-unit-set-aliases/ref",
        json={"custom_unit_set_id": v2, "expected_id": v1},
    ).status_code == 200
    resp = patch_step(
        wid, 1, "a", {"name": "a", "formula": "x*2", "output_unit": "m"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"][0]["value"] == "0.6096"


# ---------------------------------------------------------------------------
# 持久化：容器重建（进程内存储重建）后仍可读取并继续修订
# ---------------------------------------------------------------------------


def test_worksheet_survives_store_reload_and_can_keep_revising(tmp_path):
    data_dir = str(tmp_path / "data")
    store = revisions.init_store(data_dir)
    set_id = publish(FOOT)
    resp = create(
        [ev_step("a", "x", "m"), ev_step("b", "a", "ft")],
        {"x": var("10", "ft")},
        set_id=set_id,
    )
    assert resp.status_code == 201
    wid = resp.json()["id"]
    assert patch_var(wid, 1, "x", "20", "ft").status_code == 200

    # 模拟容器重建：丢弃进程内单例后从同一数据目录重新加载
    reloaded = revisions.init_store(data_dir)
    assert reloaded.count() == 1
    snapshot = reloaded.snapshot(wid)
    assert snapshot.revision == 2
    assert snapshot.custom_unit_set_id == set_id
    assert snapshot.variables["x"] == ("20", "ft")
    assert [r["value"] for r in snapshot.results] == ["6.096", "20"]

    # 重建后可继续修订；修订号从 2 继续
    revisions._store = reloaded
    resp = patch_var(wid, 2, "x", "30", "ft")
    assert resp.status_code == 200, resp.text
    assert resp.json()["revision"] == 3
    assert resp.json()["results"][0]["value"] == "9.144"


def test_persisted_file_is_plain_json_with_revision_and_pinned_id(tmp_path):
    import json

    data_dir = str(tmp_path / "data")
    revisions.init_store(data_dir)
    set_id = publish(FOOT)
    wid, _ = create_simple(set_id=set_id)
    with open(f"{data_dir}/{revisions.STORE_FILENAME}", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["version"] == revisions._STORE_VERSION
    item = raw["worksheets"][wid]
    assert item["revision"] == 1
    assert item["custom_unit_set_id"] == set_id
    assert item["variables"]["t"] == {"value": "2", "unit": "h"}
    assert len(item["steps"]) == len(item["results"]) == 2


# ---------------------------------------------------------------------------
# 写盘失败：不改变已保存的内容、结果或修订号
# ---------------------------------------------------------------------------


def test_create_persist_failure_leaves_no_worksheet(monkeypatch):
    store = revisions.get_worksheet_store()

    def boom(self) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(revisions.RevisableWorksheetStore, "_persist_locked", boom)
    steps, variables = base_worksheet()
    resp = create(steps, variables, set_id=publish(FOOT))
    assert resp.status_code == 500
    assert store.count() == 0
    monkeypatch.undo()
    # 存储仍可正常使用
    wid, _ = create_simple()
    assert client.get(f"/revisable-worksheets/{wid}").status_code == 200


def test_revise_persist_failure_rolls_back_content_and_revision(monkeypatch):
    wid, before = create_simple()
    store = revisions.get_worksheet_store()
    calls = {"n": 0}
    real = revisions.RevisableWorksheetStore._persist_locked

    def fail_once(self) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real(self)

    monkeypatch.setattr(revisions.RevisableWorksheetStore, "_persist_locked", fail_once)
    resp = patch_var(wid, 1, "t", "3", "h")
    assert resp.status_code == 500
    monkeypatch.undo()

    # 内存中的内容、结果与修订号全部保持修订 1
    snapshot = store.snapshot(wid)
    assert snapshot.revision == 1
    assert snapshot.variables["t"] == ("2", "h")
    assert [r["value"] for r in snapshot.results] == ["120", "60"]
    # 磁盘文件也仍是旧内容（原子替换未发生）
    reloaded = revisions.RevisableWorksheetStore(store.path.rsplit("/", 1)[0])
    reloaded.load()
    assert reloaded.snapshot(wid).revision == 1
    # 回滚后可用原修订号重试并成功
    resp = patch_var(wid, 1, "t", "3", "h")
    assert resp.status_code == 200
    assert resp.json()["revision"] == 2


# ---------------------------------------------------------------------------
# 并发：同修订最多成功一项（409）；并发读取只见同一修订的输入与结果
# ---------------------------------------------------------------------------


def test_concurrent_revisions_same_base_only_one_succeeds():
    wid, _ = create_simple()
    barrier = threading.Barrier(2)
    store = revisions.get_worksheet_store()
    real_commit = store.commit_revision

    def synced_commit(*args, **kwargs):
        # 两个线程都完成整单重算后，同时进入 CAS 临界区
        barrier.wait(timeout=10)
        return real_commit(*args, **kwargs)

    outcomes: list = []

    def worker(value):
        # 直接走服务层（与端点相同的代码路径），在提交点对齐
        from app.schemas import ReviseWorksheetRequest

        req = ReviseWorksheetRequest.model_validate(
            {"expected_revision": 1, "variable": {"name": "t", "value": value, "unit": "h"}}
        )
        try:
            resp = revisions.revise_revisable_worksheet(wid, req)
            outcomes.append(("ok", resp.revision, resp.variables["t"].value))
        except revisions.RevisionConflictError as exc:
            outcomes.append(("conflict", exc.current_revision))

    # 让两个线程在 CAS 提交处对齐：在 commit_revision 外设屏障
    store.commit_revision = synced_commit  # type: ignore[method-assign]
    try:
        threads = [threading.Thread(target=worker, args=(v,)) for v in ("3", "4")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
    finally:
        store.commit_revision = real_commit  # type: ignore[method-assign]

    assert len(outcomes) == 2
    oks = [o for o in outcomes if o[0] == "ok"]
    conflicts = [o for o in outcomes if o[0] == "conflict"]
    assert len(oks) == 1
    assert len(conflicts) == 1
    assert oks[0][1] == 2
    assert conflicts[0][1] == 2
    final = client.get(f"/revisable-worksheets/{wid}").json()
    assert final["revision"] == 2
    assert final["variables"]["t"]["value"] in ("3", "4")
    expected_s = str(60 * int(final["variables"]["t"]["value"]))
    assert final["results"][0]["value"] == expected_s
    assert final["results"][1]["value"] == str(int(expected_s) // 2)


def test_concurrent_revisions_via_http_only_one_wins():
    wid, _ = create_simple()
    store = revisions.get_worksheet_store()
    start = threading.Event()
    real_commit = store.commit_revision

    def gated_commit(*args, **kwargs):
        start.set()
        return real_commit(*args, **kwargs)

    store.commit_revision = gated_commit  # type: ignore[method-assign]
    statuses: list = []
    lock = threading.Lock()

    def worker(payload):
        resp = client.patch(
            f"/revisable-worksheets/{wid}",
            json={
                "expected_revision": 1,
                "variable": {"name": "t", "value": payload, "unit": "h"},
            },
        )
        with lock:
            statuses.append(resp.status_code)

    try:
        threads = [threading.Thread(target=worker, args=(v,)) for v in ("3", "4")]
        for t in threads:
            t.start()
        assert start.wait(timeout=10)
        for t in threads:
            t.join(timeout=15)
    finally:
        store.commit_revision = real_commit  # type: ignore[method-assign]

    assert sorted(statuses) == [200, 409]
    final = client.get(f"/revisable-worksheets/{wid}").json()
    assert final["revision"] == 2


def test_reads_always_see_consistent_revision_snapshot():
    # TestClient 的传输层默认只允许单个在途请求，无法真正并发地边写边读；
    # 这里在服务层多线程并发修订与读取（与端点相同的代码路径），验证读
    # 不变式：任何一次读取的 results 必须与其 revision/variables 自洽，
    # 读者绝不会看到“新输入 + 旧结果”的跨修订拼装。
    from app.schemas import ReviseWorksheetRequest

    wid, _ = create_simple()
    store = revisions.get_worksheet_store()
    stop = threading.Event()
    bad: list = []

    def reader():
        while not stop.is_set():
            data = revisions.get_revisable_worksheet(wid, store=store)
            rev = data.revision
            hours = int(data.variables["t"].value)
            # 合法状态：修订 1 输入 2 h；此后第 k 次修订（k=1..11）把输入
            # 改为 (k+1) h 并产生修订 k+1，故修订 r≥2 时输入为 r h。
            expected_hours = 2 if rev == 1 else rev
            if hours != expected_hours or not 1 <= rev <= 12:
                bad.append(("input", rev, hours))
            values = [r.value for r in data.results]
            if values != [str(60 * hours), str(30 * hours)]:
                bad.append(("results", rev, values))

    def writer():
        for expected in range(1, 12):
            req = ReviseWorksheetRequest.model_validate(
                {
                    "expected_revision": expected,
                    "variable": {"name": "t", "value": str(expected + 1), "unit": "h"},
                }
            )
            revisions.revise_revisable_worksheet(wid, req, store=store)

    rthread = threading.Thread(target=reader)
    wthread = threading.Thread(target=writer)
    rthread.start()
    wthread.start()
    wthread.join(timeout=20)
    stop.set()
    rthread.join(timeout=10)
    assert not wthread.is_alive() and not rthread.is_alive()
    assert bad == []
    final = client.get(f"/revisable-worksheets/{wid}").json()
    assert final["revision"] == 12
    assert [r["value"] for r in final["results"]] == ["720", "360"]


def test_readers_during_revision_see_only_old_or_new_revision():
    # 修订提交（CAS + 落盘）期间的读取只能落在提交前或提交后：提交临界区
    # 持锁写入时，读者全部被挡在锁外，只能读到旧修订；放行后写入完成，
    # 之后的读取才看到新修订，绝不会读到半成品。
    wid, _ = create_simple()
    store = revisions.get_worksheet_store()
    persist_entered = threading.Event()
    release_persist = threading.Event()
    real_persist = store._persist_locked
    reads_after_release: list = []

    def gated_persist():
        persist_entered.set()
        assert release_persist.wait(timeout=10)
        return real_persist()

    def reader():
        # 进入后先阻塞在存储锁（写入正持锁）；放行后完成的读取全部收集起来
        for _ in range(20):
            data = revisions.get_revisable_worksheet(wid, store=store)
            reads_after_release.append(
                (data.revision, data.variables["t"].value, data.results[0].value)
            )

    def gatekeeper():
        # 等待写线程进入落盘临界区（持有存储锁、阻塞在 release_persist 上）
        assert persist_entered.wait(timeout=10)
        rthread = threading.Thread(target=reader)
        rthread.start()
        # 稍等读者进入并阻塞在同一把存储锁上：写入持锁期间它无法完成读取
        threading.Event().wait(0.2)
        assert rthread.is_alive(), "写入持锁期间读者不应完成任何读取"
        release_persist.set()
        rthread.join(timeout=10)

    store._persist_locked = gated_persist  # type: ignore[method-assign]
    try:
        keeper = threading.Thread(target=gatekeeper)
        keeper.start()
        resp = patch_var(wid, 1, "t", "3", "h")
        keeper.join(timeout=15)
    finally:
        store._persist_locked = real_persist  # type: ignore[method-assign]

    assert resp.status_code == 200, resp.text
    # 读者在写入持锁期间无法完成读取；临界区释放后读到的只能是完整的新修订
    assert reads_after_release, "放行后读者应完成读取"
    assert set(reads_after_release) == {(2, "3", "180")}
    final = client.get(f"/revisable-worksheets/{wid}").json()
    assert final["revision"] == 2
    assert final["variables"]["t"]["value"] == "3"
    assert final["results"][0]["value"] == "180"


# ---------------------------------------------------------------------------
# 与固定定义集一起重算：单位解析、精确传递、温标规则沿用 /worksheet
# ---------------------------------------------------------------------------


def test_revision_reruns_under_pinned_set_definitions():
    set_id = publish(FOOT)
    resp = create(
        [ev_step("m", "x", "m"), ev_step("back", "m", "ft")],
        {"x": var("10", "ft")},
        set_id=set_id,
    )
    assert resp.status_code == 201
    wid = resp.json()["id"]
    assert [r["value"] for r in resp.json()["results"]] == ["3.048", "10"]
    resp = patch_var(wid, 1, "x", "20", "ft")
    assert resp.status_code == 200
    assert [r["value"] for r in resp.json()["results"]] == ["6.096", "20"]
    # 换成定义集中不存在的单位 -> 整单重算失败，内容不变
    resp = patch_var(wid, 2, "x", "10", "furlong")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "unknown_symbol"
    assert client.get(f"/revisable-worksheets/{wid}").json()["revision"] == 2


def test_revision_to_absolute_temperature_propagates_attribute():
    variables = {"T0": var("98.6", "degF")}
    steps = [
        ev_step("T", "T0", "degC"),
        ev_step("Tk", "T", "K"),
    ]
    resp = create(steps, variables, set_id=publish(FOOT))
    assert resp.status_code == 201
    wid = resp.json()["id"]
    assert [r["value"] for r in resp.json()["results"]] == ["37", "310.15"]
    # 修改初始温度后，两个下游步骤全部按绝对温标链路重算（32 degF -> 0 degC -> 273.15 K）
    resp = patch_var(wid, 1, "T0", "32", "degF")
    assert resp.status_code == 200, resp.text
    assert [r["value"] for r in resp.json()["results"]] == ["0", "273.15"]


def test_solve_step_revision_recomputes_downstream():
    variables = {
        "v": var("60", "km/h"),
        "s": var("120", "km"),
        "v2": var("30", "km/h"),
    }
    steps = [
        solve_step("t", "v*t = s", "t", "h"),
        ev_step("s2", "t*v2", "km"),
    ]
    resp = create(steps, variables, set_id=publish(FOOT))
    assert resp.status_code == 201
    wid = resp.json()["id"]
    assert [r["value"] for r in resp.json()["results"]] == ["2", "60"]
    # s 改为 180 km：t=3 h，下游 s2=90 km（不是旧的 60）
    resp = patch_var(wid, 1, "s", "180", "km")
    assert resp.status_code == 200
    assert [r["value"] for r in resp.json()["results"]] == ["3", "90"]


def test_no_variables_worksheet_can_revise_steps():
    # 无初始变量的工作单：只能修订步骤
    steps = [
        solve_step("x", "2*x = 10", "x", "1"),
        ev_step("y", "x*3", "1"),
    ]
    resp = create(steps, set_id=publish(FOOT))
    assert resp.status_code == 201, resp.text
    wid = resp.json()["id"]
    assert [r["value"] for r in resp.json()["results"]] == ["5", "15"]
    resp = patch_step(
        wid, 1, "x", solve_step("x", "2*x = 16", "x", "1")
    )
    assert resp.status_code == 200, resp.text
    assert [r["value"] for r in resp.json()["results"]] == ["8", "24"]
    # 没有初始变量时替换变量必然 422
    resp = patch_var(wid, 2, "nope", "1", "1")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 原有接口与 Compose 不受影响
# ---------------------------------------------------------------------------


def test_original_worksheet_endpoint_still_works():
    resp = client.post(
        "/worksheet",
        json={
            "variables": {"v": var("60", "km/h"), "t": var("2", "h")},
            "steps": [
                ev_step("s", "v*t", "km"),
                ev_step("half", "s/2", "km"),
            ],
        },
    )
    assert resp.status_code == 200
    assert [r["value"] for r in resp.json()["steps"]] == ["120", "60"]


def test_original_worksheet_still_accepts_request_level_custom_units():
    resp = client.post(
        "/worksheet",
        json={
            "variables": {"x": var("10", "ft")},
            "steps": [
                ev_step("a", "x", "m"),
                ev_step("b", "a", "ft"),
            ],
            "custom_units": [FOOT],
        },
    )
    assert resp.status_code == 200
