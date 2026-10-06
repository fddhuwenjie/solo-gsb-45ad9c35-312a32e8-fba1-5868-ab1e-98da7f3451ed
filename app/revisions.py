"""可修订工作单（revisable worksheets）。

工作单需要反复调整输入或步骤，并保留每次成功修改后的整单结果：

- ``POST /revisable-worksheets`` 提交一份现有的 2～5 步工作单（结构与
  ``POST /worksheet`` 一致）与**已发布定义集标识或活动别名**。别名只在
  创建请求开始时解析一次，实际定义集标识随工作单固定保存——此后别名再
  被切换，已创建的工作单仍始终使用创建时的那一版定义。**整单计算成功后**
  才把输入、逐步结果、固定的定义集标识与初始修订号（1）共同保存，并返回
  服务生成的不可猜测标识；计算失败（422）时不保存任何内容。
- ``GET /revisable-worksheets/{id}`` 按标识读取当前修订的完整内容：
  输入（变量与步骤）、逐步结果与修订号。读取在同一把锁的同一份快照内
  完成，并发读取只能看到同一修订的输入与结果，不会看到跨修订的拼装。
- ``PATCH /revisable-worksheets/{id}`` 提交 ``expected_revision`` 做
  乐观并发：一次只能替换**一个现有初始变量**的值与单位，或**一个现有
  步骤**的公式、方程及目标单位（步骤名称不变）。先在固定定义下把修改后
  的整张工作单**从头完整重算**（下游步骤一律基于新输入重新计算，绝不沿用
  旧结果），成功后才在同一把锁内做修订号 CAS、把新输入、新逐步结果与
  递增的修订号共同保存；计算失败（422）不改变已保存的任何内容，修订号
  也不递增。同一修订的并发修改最多成功一项，其余返回 **409**。

内存更新与 JSON 原子落盘在同一临界区内**共同生效**：写盘失败时回滚
本次新增/修改，已保存的内容、结果与修订号保持原样。工作单保存在数据
目录（默认 ``/data``）的 ``revisable_worksheets.json`` 中，容器重建/
进程重启后仍可读取并继续修订。
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import dataclass
from decimal import Decimal

from .config import DATA_DIR
from .errors import (
    ConversionError,
    PersistenceError,
    RevisionConflictError,
    WorksheetNotFoundError,
)
from .schemas import (
    CreateRevisableWorksheetRequest,
    RevisableVariable,
    ReviseWorksheetRequest,
    RevisableWorksheetResponse,
    Variable,
    WorksheetRequest,
    WorksheetStep,
    WorksheetStepResult,
)
from .unit_sets import get_store
from .worksheet import run_worksheet

#: 持久化文件名（位于 :data:`~app.config.DATA_DIR` 下）。
STORE_FILENAME = "revisable_worksheets.json"

#: 标识前缀（revisable worksheet）+ 随机字节数。
_ID_PREFIX = "rws_"
_ID_BYTES = 16

#: 存储文件格式版本。
_STORE_VERSION = 1

#: 初始修订号。
INITIAL_REVISION = 1


@dataclass(frozen=True)
class StoredWorksheet:
    """一份已保存的可修订工作单（不可变；修订即整体替换为新版本）。"""

    worksheet_id: str
    revision: int
    #: 创建时固定的实际定义集标识（别名只在创建时解析一次）
    custom_unit_set_id: str
    #: 变量名 -> (精确十进制字符串, 单位表达式)
    variables: dict[str, tuple[str, str]]
    #: 与提交顺序一致的步骤定义（model_dump 后的 JSON 兼容结构）
    steps: list[dict]
    #: 与步骤一一对应的逐步结果（model_dump 后的 JSON 兼容结构）
    results: list[dict]


# ---------------------------------------------------------------------------
# 序列化与请求组装
# ---------------------------------------------------------------------------


def _body_to_request(
    variables: dict[str, tuple[str, str]], steps: list[dict], set_id: str
) -> WorksheetRequest:
    """用保存的输入组装一份可交给 :func:`run_worksheet` 的工作单请求。

    始终引用固定的定义集**标识**（创建后别名切换不影响工作单）。
    """

    return WorksheetRequest(
        variables={
            name: Variable(value=Decimal(value), unit=unit)
            for name, (value, unit) in variables.items()
        },
        steps=[WorksheetStep.model_validate(item) for item in steps],
        custom_unit_set_id=set_id,
    )


def _request_from_create(req: CreateRevisableWorksheetRequest, set_id: str) -> WorksheetRequest:
    """把创建请求体转为执行用工作单请求（带固定的定义集标识）。"""

    return WorksheetRequest(
        variables={
            name: Variable(value=var.value, unit=var.unit)
            for name, var in req.variables.items()
        },
        steps=list(req.steps),
        custom_unit_set_id=set_id,
    )


def _dump_step(step: WorksheetStep) -> dict:
    """步骤序列化为 JSON 兼容结构（求值/反求字段按其有无省略）。"""

    return step.model_dump(exclude_none=True)


def _build_response(stored: StoredWorksheet) -> RevisableWorksheetResponse:
    """把保存的工作单组装为对外响应（输入、结果、修订号属同一快照）。"""

    return RevisableWorksheetResponse(
        id=stored.worksheet_id,
        revision=stored.revision,
        custom_unit_set_id=stored.custom_unit_set_id,
        variables={
            name: RevisableVariable(value=value, unit=unit)
            for name, (value, unit) in sorted(stored.variables.items())
        },
        steps=[WorksheetStep.model_validate(item) for item in stored.steps],
        results=[WorksheetStepResult.model_validate(item) for item in stored.results],
    )


# ---------------------------------------------------------------------------
# 修订应用（纯数据变换；非法请求在此抛 ConversionError 422）
# ---------------------------------------------------------------------------


def _apply_revision(
    stored: StoredWorksheet, req: ReviseWorksheetRequest
) -> tuple[dict[str, tuple[str, str]], list[dict]]:
    """按修订请求构造新输入：恰好替换一个变量或一个步骤（名称均不变）。

    目标变量/步骤必须已存在；步骤替换不得改名，也不能改变步骤数量或顺序。
    任何非法情况都抛 :class:`~app.errors.ConversionError`（422），此时不
    触碰已保存内容。
    """

    variables = dict(stored.variables)
    steps = [dict(step) for step in stored.steps]

    if req.variable is not None:
        change = req.variable
        if change.name not in variables:
            raise ConversionError(
                "invalid_request",
                f"修订只能替换现有初始变量：变量 {change.name!r} 不存在"
                "（不能新增变量、也不能改名）",
            )
        variables[change.name] = (str(change.value), change.unit)
        return variables, steps

    assert req.step is not None
    target = req.step.name
    for index, item in enumerate(steps):
        if item.get("name") == target:
            # 完整重建该步骤：名称不变，但公式/方程与目标单位全部以提交内容为准
            steps[index] = _dump_step(req.step.step)
            return variables, steps
    raise ConversionError(
        "invalid_request",
        f"修订只能替换现有步骤：步骤 {target!r} 不存在（不能新增步骤、也不能改名）",
    )


# ---------------------------------------------------------------------------
# 持久化存储
# ---------------------------------------------------------------------------


class RevisableWorksheetStore:
    """可修订工作单的持久化存储（线程安全、原子落盘、CAS 修订）。"""

    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self._data_dir = data_dir
        self._path = os.path.join(data_dir, STORE_FILENAME)
        self._lock = threading.RLock()
        self._sheets: dict[str, StoredWorksheet] = {}
        self._loaded = False

    @property
    def path(self) -> str:
        """持久化文件的完整路径。"""

        return self._path

    @property
    def lock(self) -> threading.RLock:
        """存储级临界区（测试中用于编排并发修订与读取）。"""

        return self._lock

    def load(self) -> None:
        """从数据目录加载已保存的工作单；文件不存在时以空存储启动。

        文件存在但无法解析时抛出异常使启动失败（宁可不可用也不静默丢数据）。
        """

        with self._lock:
            if not os.path.exists(self._path):
                self._sheets = {}
                self._loaded = True
                return
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict) or raw.get("version") != _STORE_VERSION:
                raise ValueError(f"可修订工作单存储文件版本不被支持: {self._path}")
            sheets: dict[str, StoredWorksheet] = {}
            raw_sheets = raw.get("worksheets", {})
            if not isinstance(raw_sheets, dict):
                raise ValueError(
                    f"可修订工作单存储文件的 worksheets 字段格式非法: {self._path}"
                )
            for sheet_id, item in raw_sheets.items():
                variables_raw = item.get("variables", {})
                if not isinstance(variables_raw, dict):
                    raise ValueError(f"工作单 {sheet_id!r} 的 variables 字段非法")
                variables: dict[str, tuple[str, str]] = {}
                for name, var in variables_raw.items():
                    if (
                        not isinstance(var, dict)
                        or not isinstance(var.get("value"), str)
                        or not isinstance(var.get("unit"), str)
                    ):
                        raise ValueError(f"工作单 {sheet_id!r} 的变量 {name!r} 非法")
                    variables[name] = (var["value"], var["unit"])
                revision = item.get("revision")
                set_id = item.get("custom_unit_set_id")
                steps = item.get("steps")
                results = item.get("results")
                if not isinstance(revision, int) or revision < 1:
                    raise ValueError(f"工作单 {sheet_id!r} 的 revision 非法")
                if not isinstance(set_id, str) or not set_id:
                    raise ValueError(f"工作单 {sheet_id!r} 的 custom_unit_set_id 非法")
                if not isinstance(steps, list) or not isinstance(results, list):
                    raise ValueError(f"工作单 {sheet_id!r} 的 steps/results 非法")
                if len(steps) != len(results):
                    raise ValueError(
                        f"工作单 {sheet_id!r} 的 steps 与 results 数量不一致"
                    )
                sheets[sheet_id] = StoredWorksheet(
                    worksheet_id=sheet_id,
                    revision=revision,
                    custom_unit_set_id=set_id,
                    variables=variables,
                    steps=steps,
                    results=results,
                )
            self._sheets = sheets
            self._loaded = True

    def create(
        self,
        *,
        set_id: str,
        variables: dict[str, tuple[str, str]],
        steps: list[dict],
        results: list[WorksheetStepResult],
    ) -> StoredWorksheet:
        """生成标识、以修订号 1 保存新工作单并与落盘共同生效。

        调用方须已完成整单计算；内存写入与原子写盘在同一临界区内完成，
        写盘失败时移除新增项后重新抛出，不留下只存在于内存的工作单。
        返回的是内容的深拷贝快照，调用方无法再修改存储内状态。
        """

        self._ensure_loaded()
        with self._lock:
            while True:
                worksheet_id = _ID_PREFIX + secrets.token_hex(_ID_BYTES)
                if worksheet_id not in self._sheets:
                    break
            stored = StoredWorksheet(
                worksheet_id=worksheet_id,
                revision=INITIAL_REVISION,
                custom_unit_set_id=set_id,
                variables=dict(variables),
                steps=[dict(step) for step in steps],
                results=[result.model_dump() for result in results],
            )
            self._sheets[worksheet_id] = stored
            try:
                self._persist_locked()
            except Exception as exc:
                # 写盘失败：移除新增工作单，创建与落盘必须共同生效
                del self._sheets[worksheet_id]
                raise PersistenceError(
                    f"可修订工作单 {worksheet_id!r} 保存失败（写盘错误）：{exc}"
                ) from exc
            return self._copy_locked(stored)

    def commit_revision(
        self,
        worksheet_id: str,
        expected_revision: int,
        *,
        variables: dict[str, tuple[str, str]],
        steps: list[dict],
        results: list[WorksheetStepResult],
    ) -> StoredWorksheet:
        """以 CAS 方式提交一次成功修订（调用方须已完成整单重算）。

        - 工作单不存在：抛 :class:`~app.errors.WorksheetNotFoundError`（404）；
        - 当前修订号与 ``expected_revision`` 不一致：抛
          :class:`~app.errors.RevisionConflictError`（409），不改变任何内容；
        - 一致：以递增后的修订号把新输入、新结果整体替换旧版本，并与原子
          落盘共同生效；写盘失败时恢复旧版本后重新抛出。
        """

        self._ensure_loaded()
        with self._lock:
            current = self._sheets.get(worksheet_id)
            if current is None:
                raise WorksheetNotFoundError(worksheet_id)
            if current.revision != expected_revision:
                raise RevisionConflictError(
                    worksheet_id, current.revision, expected_revision
                )
            updated = StoredWorksheet(
                worksheet_id=worksheet_id,
                revision=expected_revision + 1,
                custom_unit_set_id=current.custom_unit_set_id,
                variables=dict(variables),
                steps=[dict(step) for step in steps],
                results=[result.model_dump() for result in results],
            )
            self._sheets[worksheet_id] = updated
            try:
                self._persist_locked()
            except Exception as exc:
                # 写盘失败：恢复旧版本，已保存内容、结果与修订号保持原样
                self._sheets[worksheet_id] = current
                raise PersistenceError(
                    f"可修订工作单 {worksheet_id!r} 的修订 {expected_revision + 1} "
                    f"保存失败（写盘错误）：{exc}"
                ) from exc
            return self._copy_locked(updated)

    def current_revision(self, worksheet_id: str) -> int:
        """返回当前修订号；工作单不存在抛 404。（供修订前的快速冲突判定）"""

        self._ensure_loaded()
        with self._lock:
            current = self._sheets.get(worksheet_id)
            if current is None:
                raise WorksheetNotFoundError(worksheet_id)
            return current.revision

    def snapshot(self, worksheet_id: str) -> StoredWorksheet:
        """返回工作单当前修订的不可变深拷贝快照；不存在抛 404。

        输入（变量/步骤）与结果在同一临界区内取自同一版本，调用方拿到后
        即使面对并发修订，看到的也始终是同一修订的一致内容。
        """

        self._ensure_loaded()
        with self._lock:
            current = self._sheets.get(worksheet_id)
            if current is None:
                raise WorksheetNotFoundError(worksheet_id)
            return self._copy_locked(current)

    def count(self) -> int:
        """已保存的工作单数量（主要供测试与运维观察）。"""

        self._ensure_loaded()
        with self._lock:
            return len(self._sheets)

    # -- 内部 -------------------------------------------------------------

    @staticmethod
    def _copy_locked(stored: StoredWorksheet) -> StoredWorksheet:
        return StoredWorksheet(
            worksheet_id=stored.worksheet_id,
            revision=stored.revision,
            custom_unit_set_id=stored.custom_unit_set_id,
            variables=dict(stored.variables),
            steps=[dict(step) for step in stored.steps],
            results=[dict(result) for result in stored.results],
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _persist_locked(self) -> None:
        """把全部工作单全量写入临时文件并原子替换（调用方须持锁）。"""

        os.makedirs(self._data_dir, exist_ok=True)
        payload = {
            "version": _STORE_VERSION,
            "worksheets": {
                worksheet_id: {
                    "revision": stored.revision,
                    "custom_unit_set_id": stored.custom_unit_set_id,
                    "variables": {
                        name: {"value": value, "unit": unit}
                        for name, (value, unit) in stored.variables.items()
                    },
                    "steps": stored.steps,
                    "results": stored.results,
                }
                for worksheet_id, stored in sorted(self._sheets.items())
            },
        }
        tmp_path = f"{self._path}.tmp.{os.getpid()}.{secrets.token_hex(4)}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._path)


#: 进程内共享的存储单例；由应用启动时初始化（也可在测试中重新初始化）。
_store: RevisableWorksheetStore | None = None
_store_lock = threading.RLock()


def init_store(data_dir: str = DATA_DIR) -> RevisableWorksheetStore:
    """（重新）初始化进程级存储单例并加载磁盘上已保存的工作单。"""

    global _store
    with _store_lock:
        store = RevisableWorksheetStore(data_dir)
        store.load()
        _store = store
    return store


def get_worksheet_store() -> RevisableWorksheetStore:
    """获取进程级存储单例；尚未初始化时按默认数据目录初始化。"""

    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                init_store()
    assert _store is not None
    return _store


# ---------------------------------------------------------------------------
# 服务层：创建 / 读取 / 修订
# ---------------------------------------------------------------------------


def _saved_variables(
    variables: dict[str, Variable],
) -> dict[str, tuple[str, str]]:
    return {name: (str(var.value), var.unit) for name, var in variables.items()}


def create_revisable_worksheet(
    req: CreateRevisableWorksheetRequest,
    *,
    store: RevisableWorksheetStore | None = None,
    unit_sets_store=None,
) -> RevisableWorksheetResponse:
    """创建可修订工作单：固定定义集标识、整单计算成功后才保存（修订号为 1）。

    别名在请求开始时解析一次，固定下来的实际标识随工作单保存；任一步骤
    计算失败抛 :class:`~app.errors.ConversionError`（422），不生成标识、
    不保存任何内容。
    """

    chosen_sets = unit_sets_store or get_store()
    worksheets = store or get_worksheet_store()

    # 请求开始时固定实际定义集标识（id 原样使用；别名此刻解析一次）
    if req.custom_unit_set_id is not None:
        set_id = req.custom_unit_set_id
    else:
        set_id = chosen_sets.get_alias(req.custom_unit_set_alias)

    # 整单计算（此时尚未做任何保存）：任一步失败抛 422，不留任何痕迹
    sheet_req = _request_from_create(req, set_id)
    scope = chosen_sets.build_scope(set_id)
    results = run_worksheet(sheet_req, scope)

    stored = worksheets.create(
        set_id=set_id,
        variables=_saved_variables(sheet_req.variables),
        steps=[_dump_step(step) for step in sheet_req.steps],
        results=results,
    )
    return _build_response(stored)


def get_revisable_worksheet(
    worksheet_id: str, *, store: RevisableWorksheetStore | None = None
) -> RevisableWorksheetResponse:
    """按标识读取当前修订的完整内容（同一快照的输入、结果与修订号）。"""

    worksheets = store or get_worksheet_store()
    return _build_response(worksheets.snapshot(worksheet_id))


def revise_revisable_worksheet(
    worksheet_id: str,
    req: ReviseWorksheetRequest,
    *,
    store: RevisableWorksheetStore | None = None,
) -> RevisableWorksheetResponse:
    """修订可修订工作单：整单重算成功后以 CAS 共同保存新输入、结果与新修订号。

    1. 工作单不存在 -> 404；当前修订号与 ``expected_revision`` 不一致 -> 409，
       均不执行计算、不改变任何已保存内容；
    2. 应用恰好一处替换（目标必须存在、步骤不得改名），非法 -> 422；
    3. 用创建时固定的定义集把修改后的整张工作单从头完整重算——下游步骤
       一律基于新输入重新计算，绝不沿用旧结果；失败 -> 422，已保存内容、
       结果与修订号保持原样；
    4. 再次在锁内做 CAS（覆盖重算期间的并发修订）：成功则把新输入、新结果
       与递增修订号原子替换并共同落盘；冲突 -> 409；写盘失败回滚旧版本。
    """

    worksheets = store or get_worksheet_store()

    # 快速路径：不存在/修订号不符时直接失败，不做无谓的整单重算
    worksheets.current_revision(worksheet_id)

    current = worksheets.snapshot(worksheet_id)
    variables, steps = _apply_revision(current, req)

    # 整单重算（锁外、不触碰存储）：固定定义集 + 新输入从头执行全部步骤
    sheet_req = _body_to_request(variables, steps, current.custom_unit_set_id)
    scope = get_store().build_scope(current.custom_unit_set_id)
    results = run_worksheet(sheet_req, scope)

    # CAS 提交：同一修订的并发修改只有一项能在此成功，其余得到 409；
    # 内存替换与原子落盘共同生效，写盘失败由存储层回滚旧版本。
    stored = worksheets.commit_revision(
        worksheet_id,
        req.expected_revision,
        variables=_saved_variables(sheet_req.variables),
        steps=[_dump_step(step) for step in sheet_req.steps],
        results=results,
    )
    return _build_response(stored)
