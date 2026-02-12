# pysi/core/hooks/core.py

# 使い方の要点：
# run_once(cfg) の先頭で bus = HookBus(logger=logger) を作成 → set_global(bus) を呼ぶ → その後にプラグインをロード。
# これにより、デコレータ方式（@action/@filter）と register(bus)方式のプラグインが同じBusに集まります。


# pysi/core/hooks/core.py

# 使い方の要点：
# run_once(cfg) の先頭で bus = HookBus(logger=logger) を作成 → set_global(bus) を呼ぶ → その後にプラグインをロード。
# これにより、デコレータ方式（@action/@filter）と register(bus)方式のプラグインが同じBusに集まります。

from __future__ import annotations
import importlib, pkgutil, traceback, sys
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

# ---- HookBus ---------------------------------------------------------------

Callback = Callable[..., Any]

@dataclass(order=True)
class _CB:
    priority: int
    fn: Callback = field(compare=False)
    src: str = field(compare=False)

class HookBus:
    """WordPress風の Action/Filter を極小実装。例外は握り潰してコアを止めない。"""
    def __init__(self, logger=None) -> None:
        self._actions: Dict[str, List[_CB]] = {}
        self._filters: Dict[str, List[_CB]] = {}
        self.logger = logger

    # -- register -------------------------------------------------------------
    def add_action(self, name: str, fn: Callback, priority: int = 50) -> None:
        self._actions.setdefault(name, []).append(_CB(priority, fn, _src_of(fn)))
        self._actions[name].sort()

    def add_filter(self, name: str, fn: Callback, priority: int = 50) -> None:
        self._filters.setdefault(name, []).append(_CB(priority, fn, _src_of(fn)))
        self._filters[name].sort()

    # -- run ------------------------------------------------------------------
    def do_action(self, name: str, **ctx: Any) -> None:
        """
        アクションを優先度順に実行。
        - AttributeError（典型: obj.get が無い等）は犯人プラグイン特定のため詳細ログを出し、再送出して早期に気付けるようにする
        - その他の例外は従来どおり握り潰してコア継続（ログは出力）
        """
        for cb in sorted(self._actions.get(name, [])):
            try:
                cb.fn(**ctx)
            except AttributeError as e:
                # ここで犯人を特定できる詳細トレースを出す
                print(
                    f"[HOOK ERROR] action={name} plugin="
                    f"{getattr(cb.fn, '__module__', '?')}.{getattr(cb.fn, '__name__', '?')}"
                )
                traceback.print_exc()
                # AttributeError は再送出して落として原因を表面化
                raise
            except Exception:
                # それ以外は従来どおりログして継続
                _print_exc(f"[hooks] action '{name}' failed in {cb.src}", logger=self.logger)

    def apply_filters_OLD(self, name: str, value: Any, **ctx: Any) -> Any:
        out = value
        for cb in self._filters.get(name, []):
            try:
                out = cb.fn(out, **ctx)
            except Exception:
                _print_exc(f"[hooks] filter '{name}' failed in {cb.src}", logger=self.logger)
        return out



    def apply_filters(self, name: str, value: Any, **ctx: Any) -> Any:
        out = value
        for cb in self._filters.get(name, []):
            try:
                out = cb.fn(out, **ctx)
            except AttributeError:
                # ここで犯人を特定
                print(
                    f"[HOOK ERROR] filter={name} plugin="
                    f"{getattr(cb.fn, '__module__', '?')}.{getattr(cb.fn, '__name__', '?')}"
                )
                traceback.print_exc()
                # AttributeError は再送出（早く直すべき型の不一致）
                raise
            except Exception:
                _print_exc(f"[hooks] filter '{name}' failed in {cb.src}", logger=self.logger)
        return out





# ---- decorators ------------------------------------------------------------

def action(name: str, priority: int = 50) -> Callable[[Callback], Callback]:
    def _wrap(fn: Callback) -> Callback:
        hooks.add_action(name, fn, priority=priority)
        return fn
    return _wrap

def filter(name: str, priority: int = 50) -> Callable[[Callback], Callback]:
    def _wrap(fn: Callback) -> Callback:
        hooks.add_filter(name, fn, priority=priority)
        return fn
    return _wrap

# ---- loader ----------------------------------------------------------------

def autoload_plugins(package: str = "pysi.plugins") -> None:
    """pysi.plugins 以下を再帰的に import して register 実行（サブディレクトリ対応版）"""
    try:
        pkg = importlib.import_module(package)
    except Exception:
        _print_exc(f"[hooks] cannot import package: {package}", logger=getattr(hooks, "logger", None))
        return
    for _, name, is_pkg in pkgutil.walk_packages(pkg.__path__, prefix=package + "."):
        if is_pkg:
            continue
        try:
            mod = importlib.import_module(name)
            print(f"[hooks] loaded plugin: {name}")
        except Exception:
            _print_exc(f"[hooks] failed to import: {name}", logger=getattr(hooks, "logger", None))

def call_register_if_present(package: str, bus: HookBus) -> None:
    try:
        pkg = importlib.import_module(package)
    except Exception:
        logger.exception("call_register_if_present: cannot import plugin package %s", package)
        return

    for _, name, is_pkg in pkgutil.walk_packages(pkg.__path__, prefix=package + "."):
        if is_pkg:
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            logger.exception("Failed to import plugin module %s", name)
            continue

        reg = getattr(mod, "register", None)
        if callable(reg):
            try:
                reg(bus)
                logger.info("[plugins] called register(bus) on %s", name)
            except Exception:
                logger.exception("[plugins] register(bus) failed for %s", name)

# ---- utils -----------------------------------------------------------------

def _src_of(fn: Callback) -> str:
    mod = getattr(fn, "__module__", "?")
    name = getattr(fn, "__name__", "?")
    return f"{mod}:{name}"

def _print_exc(msg: str, logger=None) -> None:
    if logger is not None:
        logger.exception(msg)
    else:
        print(msg, file=sys.stderr)
        traceback.print_exc()

# global singleton
hooks = HookBus()

def set_global(bus: HookBus) -> None:
    """
    ランタイムでグローバル 'hooks' を差し替える。
    run_once(cfg) などで生成した HookBus をここに渡すと、
    @action/@filter デコレータも同じ Bus に登録される。
    """
    global hooks
    hooks = bus

# Aliases for simplicity (matching your import: add_hook, trigger_hook)
add_hook = hooks.add_action  # Use add_action as add_hook for actions
trigger_hook = hooks.do_action  # Use do_action as trigger_hook