"""재사용 가능한 데코레이터: timing(벤치마크용 시간 측정)과 caching(같은
프레임 재추론 방지).

원본 스크립트는 추론 시간을 전혀 측정하지 않았고, 평가 로직을 다시 돌릴
때마다 GPU 추론을 처음부터 다시 했다. 이 두 데코레이터를 메서드에 겹쳐
적용하면, 캐시에 있는 프레임은 추론을 건너뛰면서 그 사실이 타이밍 기록에도
그대로 반영된다 (캐시 히트 시 시간이 거의 0으로 찍힘).
"""
from __future__ import annotations
import functools
import time


def timed(label: str):
    """호출마다 걸린 시간을 self.timings[label] (list)에 기록한다.
    데코레이트되는 메서드를 가진 객체는 `timings: dict[str, list[float]]`
    속성을 가지고 있어야 한다."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            t0 = time.perf_counter()
            result = fn(self, *args, **kwargs)
            self.timings[label].append(time.perf_counter() - t0)
            return result

        return wrapper

    return deco


def cached_by_arg(cache_attr: str):
    """첫 번째 위치 인자(예: 이미지 경로)를 키로 결과를 캐싱한다.
    캐시는 self.<cache_attr> (dict)에 저장된다.

    timed와 함께 쌓아서 쓰면 (@cached_by_arg 바깥, @timed 안쪽) 캐시 히트일
    때는 timed의 본문이 실행되지 않으므로, 타이밍 기록은 실제 GPU 추론이
    일어난 경우만 포함한다 -> 캐싱 효과를 시간 측정값으로 직접 증명 가능."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, key, *args, **kwargs):
            cache = getattr(self, cache_attr)
            if key in cache:
                return cache[key]
            result = fn(self, key, *args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return deco
