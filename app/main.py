"""FastAPI 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from fractions import Fraction

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .converter import (
    convert,
    convert_fraction,
    is_absolute_target,
    to_decimal_str,
)
from .custom_units import scope_for_request
from .diagnose import diagnose_worksheet
from .errors import (
    AliasConflictError,
    ConversionError,
    MigrationError,
    PersistenceError,
    RevisionConflictError,
    WorksheetNotFoundError,
    affine_mixing,
)
from .evaluator import evaluate, make_quantity
from .migrate import migrate_worksheet
from .parser import ParsedExpression, canonical, canonical_dim, parse_expression
from .revisions import (
    create_revisable_worksheet,
    get_worksheet_store,
    get_revisable_worksheet,
    revise_revisable_worksheet,
)
from .schemas import (
    BindUnitSetAliasRequest,
    ConvertRequest,
    ConvertResponse,
    CreateRevisableWorksheetRequest,
    ErrorResponse,
    EvaluateRequest,
    PublishUnitSetRequest,
    PublishedUnitSetResponse,
    RevisableWorksheetResponse,
    ReviseWorksheetRequest,
    SolveRequest,
    UnitSetAliasInfo,
    UnitSetAliasListResponse,
    UnitSetMigrationExportRequest,
    UnitSetMigrationImportResponse,
    UnitSetMigrationPackage,
    WorksheetDiagnoseRequest,
    WorksheetDiagnoseResponse,
    WorksheetMigrateRequest,
    WorksheetMigrateResponse,
    WorksheetRequest,
    WorksheetResponse,
)
from .solver import solve
from .unit_sets import get_store
from .units import ALIASES, PREFIXES, REGISTRY
from .worksheet import run_worksheet


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 启动时加载磁盘上的已发布自定义单位定义集与可修订工作单（容器重建后仍可使用）
    get_store()
    get_worksheet_store()
    yield


app = FastAPI(
    title="量纲表达式与仿射单位换算 API",
    version=__version__,
    description=(
        "解析单位表达式（乘、除、整数幂），规范化量纲，支持 SI 词头与"
        "摄氏/华氏/开尔文仿射温标；/evaluate 支持带单位的变量公式求值，"
        "/solve 支持用带单位的等式反求一个未知量（未知量须处于一次线性位置），"
        "所有结果均为精确十进制，维度不兼容、未知符号、非法幂、除零、"
        "非法仿射用法、非线性方程、零系数或精度溢出时返回 422。"
        "/worksheet 可在一次请求内顺序执行 2～5 个求值/反求步骤，"
        "后续步骤按结果名引用先前步骤，步骤间传递精确数值与绝对温度属性。"
        "/worksheet-diagnose 把同一工作单分别放到旧/新两份已发布定义集"
        "（标识或活动别名）下各执行一遍，按内部精确数值、量纲与绝对温度"
        "属性逐步对照并报告首个含义变化。"
        "/worksheet-migrate 则把现有工作单的初始变量改写为在新定义集下"
        "物理量相同的精确十进制值，返回可直接交给 /worksheet 的新工作单。"
        "POST/GET/PATCH /revisable-worksheets 提供可反复修订、持久保存的"
        "工作单：创建时固定实际定义集标识，整单成功后保存输入、逐步结果"
        "与修订号；修订须带 expected_revision，整单重算成功后共同保存新"
        "输入、新结果与递增修订号，冲突返回 409、计算失败返回 422 且不"
        "改变已保存内容。"
        "可通过 POST /custom-unit-sets 把自定义单位发布为可复用定义集，"
        "再由四个接口凭标识选用。"
        "POST /custom-unit-set-migration/export 可一次导出 1～6 个已发布"
        "定义集及指向它们的活动别名作为迁移包，另一实例用 "
        "POST /custom-unit-set-migration/import 整包校验后导入：为各定义集"
        "生成新本地标识、迁移别名指向并返回旧→新标识映射。"
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 异常处理：领域错误与参数校验错误统一返回 422
# ---------------------------------------------------------------------------


@app.exception_handler(ConversionError)
async def conversion_error_handler(_: Request, exc: ConversionError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": {"error": exc.code, "message": exc.message}},
    )


@app.exception_handler(MigrationError)
async def migration_error_handler(_: Request, exc: MigrationError) -> JSONResponse:
    # 迁移失败：在统一 422 负载上额外指出出错的变量或步骤，不返回迁移工作单
    detail: dict = {"error": exc.code, "message": exc.message}
    if exc.variable is not None:
        detail["variable"] = exc.variable
    if exc.step_index is not None:
        detail["step_index"] = exc.step_index
        detail["step_name"] = exc.step_name
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(AliasConflictError)
async def alias_conflict_error_handler(_: Request, exc: AliasConflictError) -> JSONResponse:
    # 乐观切换冲突：当前指向与 expected_id 不一致；不改变任何指向
    detail: dict = {"error": "alias_conflict", "message": exc.message}
    if exc.current_id is not None:
        detail["current_id"] = exc.current_id
    return JSONResponse(status_code=409, content={"detail": detail})


@app.exception_handler(RevisionConflictError)
async def revision_conflict_error_handler(
    _: Request, exc: RevisionConflictError
) -> JSONResponse:
    # 可修订工作单乐观修订冲突：同一修订的并发修改最多成功一项；
    # 不改变已保存的输入、结果与修订号，响应带回服务端当前修订号供重试
    return JSONResponse(
        status_code=409,
        content={
            "detail": {
                "error": "revision_conflict",
                "message": exc.message,
                "current_revision": exc.current_revision,
            }
        },
    )


@app.exception_handler(WorksheetNotFoundError)
async def worksheet_not_found_error_handler(
    _: Request, exc: WorksheetNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": {"error": "worksheet_not_found", "message": exc.message}},
    )


@app.exception_handler(PersistenceError)
async def persistence_error_handler(_: Request, exc: PersistenceError) -> JSONResponse:
    # 落盘失败：存储层已回滚内存状态，已保存内容、结果与修订号保持原样
    return JSONResponse(
        status_code=500,
        content={"detail": {"error": "persistence_failed", "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "invalid_request",
                "message": "请求参数校验失败",
                "errors": errors,
            }
        },
    )


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@app.post(
    "/custom-unit-sets",
    response_model=PublishedUnitSetResponse,
    status_code=201,
    responses={422: {"model": ErrorResponse, "description": "任一条目非法，未发布"}},
    summary="发布可复用的自定义单位定义集",
)
def publish_unit_set_endpoint(req: PublishUnitSetRequest) -> PublishedUnitSetResponse:
    # 先按现有规则校验并展开全部条目：任一项非法即抛 422，且不写入任何定义集
    set_id = get_store().publish(req.custom_units)
    return PublishedUnitSetResponse(
        id=set_id,
        custom_units=[
            {"symbol": item.symbol, "factor": str(item.factor), "base_unit": item.base_unit}
            for item in req.custom_units
        ],
    )


@app.get(
    "/custom-unit-set-aliases",
    response_model=UnitSetAliasListResponse,
    summary="查询全部活动别名及其当前指向的定义集",
)
def list_unit_set_aliases_endpoint() -> UnitSetAliasListResponse:
    aliases = get_store().list_aliases()
    return UnitSetAliasListResponse(
        aliases=[
            UnitSetAliasInfo(alias=alias, custom_unit_set_id=target)
            for alias, target in aliases.items()
        ]
    )


@app.get(
    "/custom-unit-set-aliases/{alias}",
    response_model=UnitSetAliasInfo,
    responses={422: {"model": ErrorResponse, "description": "别名不存在"}},
    summary="查询单个活动别名当前指向的定义集",
)
def get_unit_set_alias_endpoint(alias: str) -> UnitSetAliasInfo:
    return UnitSetAliasInfo(
        alias=alias, custom_unit_set_id=get_store().get_alias(alias)
    )


@app.put(
    "/custom-unit-set-aliases/{alias}",
    response_model=UnitSetAliasInfo,
    responses={
        409: {"description": "当前指向与 expected_id 不一致，未切换"},
        422: {"model": ErrorResponse, "description": "目标定义集不存在或别名非法，未切换"},
    },
    summary="创建/切换活动别名（乐观并发：expected_id）",
)
def bind_unit_set_alias_endpoint(
    alias: str, req: BindUnitSetAliasRequest
) -> UnitSetAliasInfo:
    # 仅当当前指向与 expected_id 一致时生效（初建须为 null）；
    # 目标不存在返回 422；切换与落盘共同生效，失败不改写原指向。
    target = get_store().bind_alias(
        alias, req.custom_unit_set_id, req.expected_id
    )
    return UnitSetAliasInfo(alias=alias, custom_unit_set_id=target)


@app.post(
    "/custom-unit-set-migration/export",
    response_model=UnitSetMigrationPackage,
    responses={422: {"model": ErrorResponse, "description": "标识不存在、重复或数量越界，未导出"}},
    summary="导出定义集迁移包（1～6 个定义集及指向它们的活动别名）",
)
def export_unit_set_migration_endpoint(
    req: UnitSetMigrationExportRequest,
) -> UnitSetMigrationPackage:
    # 定义集存在性校验与别名筛选在存储层同一把锁内的同一份快照完成：
    # 并发切换别名时，包内定义与指向必然相互吻合，不会跨快照拼装。
    sets, aliases = get_store().export_package(list(req.custom_unit_set_ids))
    return UnitSetMigrationPackage(
        sets={
            set_id: list(items)
            for set_id, items in sets.items()
        },
        aliases=aliases,
    )


@app.post(
    "/custom-unit-set-migration/import",
    response_model=UnitSetMigrationImportResponse,
    status_code=201,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "包内定义非法、别名引用悬空、别名非法或目标端已有"
            "同名别名：整包拒绝，不留下部分定义或别名",
        }
    },
    summary="导入定义集迁移包：生成新本地标识并迁移别名指向",
)
def import_unit_set_migration_endpoint(
    package: UnitSetMigrationPackage,
) -> UnitSetMigrationImportResponse:
    # 先按发布时完全相同的规则校验包内全部定义，再校验别名引用完整性与
    # 目标端同名冲突；任一失败整包拒绝。通过后新标识生成、别名指向与原子
    # 落盘在同一临界区共同生效，写盘失败回滚全部新增内容。
    id_map = get_store().import_package(
        sets={set_id: list(items) for set_id, items in package.sets.items()},
        aliases=dict(package.aliases),
    )
    return UnitSetMigrationImportResponse(
        package=package,
        id_map=id_map,
        aliases=[
            UnitSetAliasInfo(alias=alias, custom_unit_set_id=id_map[old_target])
            for alias, old_target in sorted(package.aliases.items())
        ],
    )


@app.post(
    "/convert",
    response_model=ConvertResponse,
    responses={422: {"model": ErrorResponse, "description": "换算失败"}},
    summary="单位换算",
)
def convert_endpoint(req: ConvertRequest) -> ConvertResponse:
    scope = scope_for_request(
        req.custom_units, req.custom_unit_set_id, req.custom_unit_set_alias
    )
    src = parse_expression(req.from_unit, scope)
    dst = parse_expression(req.to_unit, scope)
    value = convert(req.value, src, dst)
    return ConvertResponse(
        value=value,
        unit=canonical(dst.terms),
        dimension=canonical_dim(dst.dim),
    )


@app.post(
    "/evaluate",
    response_model=ConvertResponse,
    responses={422: {"model": ErrorResponse, "description": "求值/换算失败"}},
    summary="带单位的变量公式求值并换算到目标单位",
)
def evaluate_endpoint(req: EvaluateRequest) -> ConvertResponse:
    scope = scope_for_request(
        req.custom_units, req.custom_unit_set_id, req.custom_unit_set_alias
    )
    # 变量先解析单位并折叠为 SI 口径的精确量；未知单位在此即报 422
    quantities = {
        name: make_quantity(var.value, parse_expression(var.unit, scope))
        for name, var in req.variables.items()
    }
    result = evaluate(req.formula, quantities)
    dst = parse_expression(req.output_unit, scope)

    if result.absolute:
        # 绝对温标变量作为公式唯一项：走仿射换算路径
        if not is_absolute_target(dst):
            raise affine_mixing(
                "绝对温度结果只能换算到绝对温标（degC/degF/K），"
                "不能换算到温差单位或复合单位"
            )
        src = ParsedExpression(dim=result.dim, scale=Fraction(1), terms=(("K", 1),))
        value_fr = convert_fraction(result.value_si, src, dst)
    else:
        # 线性量（含温差）：以 SI 基单位口径构造源表达式
        src = ParsedExpression(dim=result.dim, scale=Fraction(1), terms=())
        value_fr = convert_fraction(result.value_si, src, dst)

    return ConvertResponse(
        value=to_decimal_str(value_fr),
        unit=canonical(dst.terms),
        dimension=canonical_dim(dst.dim),
    )


@app.post(
    "/solve",
    response_model=ConvertResponse,
    responses={422: {"model": ErrorResponse, "description": "求解失败"}},
    summary="用带单位的等式反求一个未知量",
)
def solve_endpoint(req: SolveRequest) -> ConvertResponse:
    scope = scope_for_request(
        req.custom_units, req.custom_unit_set_id, req.custom_unit_set_alias
    )
    # 变量先解析单位并折叠为 SI 口径的精确量；未知单位在此即报 422
    quantities = {
        name: make_quantity(var.value, parse_expression(var.unit, scope))
        for name, var in req.variables.items()
    }
    value_si, dim = solve(req.equation, req.solve_for, quantities)
    dst = parse_expression(req.output_unit, scope)
    if dst.affine is not None:
        raise affine_mixing(
            "求解结果只能换算到线性单位，不能换算到绝对温标（degC/degF）"
        )
    if dim is None:
        # 解恒为零且量纲不受方程约束（如 2*x = x）：接受任意线性目标单位
        value_fr = Fraction(0)
    else:
        # 线性量：以 SI 基单位口径构造源表达式
        src = ParsedExpression(dim=dim, scale=Fraction(1), terms=())
        value_fr = convert_fraction(value_si, src, dst)
    return ConvertResponse(
        value=to_decimal_str(value_fr),
        unit=canonical(dst.terms),
        dimension=canonical_dim(dst.dim),
    )


@app.post(
    "/worksheet",
    response_model=WorksheetResponse,
    responses={422: {"model": ErrorResponse, "description": "任一步骤失败；不返回部分结果"}},
    summary="按顺序执行的量纲计算工作单（2～5 步，步骤间可引用先前结果）",
)
def worksheet_endpoint(req: WorksheetRequest) -> WorksheetResponse:
    # 单位上下文在工作单开始时构造一次：三种自定义单位来源在此选择其一，
    # 活动别名也在此刻解析一次，随后所有步骤共用同一版定义（执行期间切换
    # 别名不影响本工作单）。任一步骤错误均抛 422，不返回部分结果。
    scope = scope_for_request(
        req.custom_units, req.custom_unit_set_id, req.custom_unit_set_alias
    )
    return WorksheetResponse(steps=run_worksheet(req, scope))


@app.post(
    "/worksheet-diagnose",
    response_model=WorksheetDiagnoseResponse,
    responses={
        422: {"model": ErrorResponse, "description": "请求非法或旧/新定义集（标识或别名）不存在"}
    },
    summary="工作单换版诊断：同一 2～5 步工作单分别在旧/新已发布定义集下执行并逐步对照",
)
def worksheet_diagnose_endpoint(
    req: WorksheetDiagnoseRequest,
) -> WorksheetDiagnoseResponse:
    # 开始时分别固定旧/新两侧引用的实际指向（别名各解析一次），随后两侧各用
    # 各的独立 scope 执行同一份工作单，不修改定义集或别名；按内部精确分数、
    # 量纲与绝对温度属性比较（而非显示字符串）。工作单业务错误（变量解析或
    # 某步失败）结构化出现在 200 响应中；未知定义集/别名与非法请求仍为 422。
    return diagnose_worksheet(req)


@app.post(
    "/worksheet-migrate",
    response_model=WorksheetMigrateResponse,
    response_model_exclude_none=True,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "旧工作单出错、变量/步骤在新定义下不兼容、改写值无法"
            "精确十进制表示、新工作单出错或结果含义变化；响应指出变量或步骤，"
            "不返回迁移工作单",
        }
    },
    summary="工作单输入迁移：把现有 2～5 步工作单改写为在新定义集下物理量"
    "相同的等价工作单",
)
def worksheet_migrate_endpoint(req: WorksheetMigrateRequest) -> WorksheetMigrateResponse:
    # 请求开始分别固定旧/新两侧引用的实际指向（别名各解析一次，执行期间
    # 切换不混版）；先在旧定义下执行原工作单，再把每个初始变量改写为
    # 新定义下同一单位表达式、同一物理量的精确十进制值；最后用固定的新
    # 标识组装新工作单并完整复核（逐步核对内部精确量、量纲与绝对温度
    # 属性，显示值允许变化）。任何一处失败都返回 422 并指出变量或步骤，
    # 不返回迁移工作单；整个操作只读，不修改定义集或别名。
    return migrate_worksheet(req)


@app.post(
    "/revisable-worksheets",
    response_model=RevisableWorksheetResponse,
    response_model_exclude_none=True,
    status_code=201,
    responses={
        422: {"model": ErrorResponse, "description": "定义集标识/别名不存在或工作单"
        "任一步骤计算失败：不保存任何内容，不生成标识"}
    },
    summary="创建可修订工作单（整单成功后保存输入、逐步结果、固定定义集标识与修订号 1）",
)
def create_revisable_worksheet_endpoint(
    req: CreateRevisableWorksheetRequest,
) -> RevisableWorksheetResponse:
    # 提交现有 2～5 步工作单与已发布定义集标识或活动别名：别名在请求开始
    # 时解析一次，实际标识随工作单固定（此后别名切换不影响本工作单）。
    # 整单计算成功后才把输入、逐步结果、实际定义集标识与初始修订号共同
    # 保存并返回工作单标识；计算失败返回 422 且不留任何内容。
    return create_revisable_worksheet(req)


@app.get(
    "/revisable-worksheets/{worksheet_id}",
    response_model=RevisableWorksheetResponse,
    response_model_exclude_none=True,
    responses={404: {"description": "工作单标识不存在"}},
    summary="按标识读取可修订工作单的当前内容、逐步结果与修订号",
)
def get_revisable_worksheet_endpoint(worksheet_id: str) -> RevisableWorksheetResponse:
    # 输入与结果取自同一临界区内的同一版本：并发读取只会看到某一修订的
    # 一致内容，不会看到“新输入 + 旧结果”的跨修订拼装。
    return get_revisable_worksheet(worksheet_id)


@app.patch(
    "/revisable-worksheets/{worksheet_id}",
    response_model=RevisableWorksheetResponse,
    response_model_exclude_none=True,
    responses={
        409: {"description": "expected_revision 与当前修订号不一致（同一修订的并发"
        "修改最多成功一项）；已保存内容不变"},
        422: {"model": ErrorResponse, "description": "替换目标不存在/改名/步骤数变化，"
        "或整单重算失败；已保存内容、结果与修订号均不变"},
        404: {"description": "工作单标识不存在"},
    },
    summary="修订可修订工作单（替换一个初始变量或一个现有步骤，整单重算后递增修订号）",
)
def revise_revisable_worksheet_endpoint(
    worksheet_id: str, req: ReviseWorksheetRequest
) -> RevisableWorksheetResponse:
    # 乐观并发：expected_revision 不符返回 409；一次只能替换一个现有初始
    # 变量的值与单位，或一个现有步骤的公式/方程及目标单位（名称不变）。
    # 修改后整张工作单用创建时固定的定义集从头完整重算（下游步骤不沿用
    # 旧结果），成功后新输入、新逐步结果与递增修订号在同一临界区内共同
    # 保存并原子落盘；计算失败 422、写盘失败回滚，都不改变已保存内容。
    return revise_revisable_worksheet(worksheet_id, req)


@app.get("/health", summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/units", summary="列出全部支持的单位、词头与别名")
def list_units() -> dict:
    return {
        "units": [
            {
                "symbol": u.symbol,
                "name": u.name,
                "dimension": canonical_dim(u.dim),
                "affine": u.affine,
                "prefixable": u.prefixable,
            }
            for u in REGISTRY.values()
        ],
        "prefixes": sorted(PREFIXES),
        "aliases": ALIASES,
    }


@app.get("/", summary="服务信息")
def root() -> dict[str, str]:
    return {
        "service": "量纲表达式与仿射单位换算 API",
        "version": __version__,
        "docs": "/docs",
    }
