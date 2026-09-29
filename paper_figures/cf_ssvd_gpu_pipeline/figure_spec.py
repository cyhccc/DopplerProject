from dataclasses import dataclass
from typing import Mapping

FIGURE_WIDTH_MM = 150.0
FIGURE_HEIGHT_MM = 66.0
PNG_DPI = 600
FORBIDDEN_FIG1_TERMS = (
    "Thread0", "Thread31", "warp", "frame lane", "shared memory",
    "线程块结构", "thread block",
)

COLORS = {
    "flow": "#344F66", "blue": "#2F73B7", "blue_strong": "#125FA8",
    "blue_light": "#EAF4FF", "purple": "#6558A8",
    "purple_light": "#EFEDFF", "warm": "#C35C35",
    "warm_light": "#FFF0E9", "gpu": "#173C5D",
}


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    label: str
    visual_kind: str


@dataclass(frozen=True)
class RectMM:
    x: float
    y: float
    width: float
    height: float


MODULES = (
    ModuleSpec("iq", "IQ解调", "iq_demodulation"),
    ModuleSpec("cpwc", "CPWC-DAS波束形成", "cpwc_das"),
    ModuleSpec("ssvd", "sSVD+SSM杂波滤波", "ssvd_ssm"),
    ModuleSpec("power", "功率累积", "power_sum"),
)


def build_layout() -> dict[str, RectMM]:
    return {
        "acquisition": RectMM(2, 8, 18, 42), "iq": RectMM(23, 10, 20, 35),
        "cpwc": RectMM(46, 10, 23, 35), "ssvd": RectMM(72, 10, 23, 35),
        "power": RectMM(98, 10, 20, 35), "output": RectMM(121, 9, 27, 36),
        "gpu": RectMM(55, 0.5, 40, 9),
    }


def validate_layout(layout: Mapping[str, RectMM]) -> None:
    for key, rect in layout.items():
        if rect.x < 0 or rect.y < 0 or rect.x + rect.width > FIGURE_WIDTH_MM or rect.y + rect.height > FIGURE_HEIGHT_MM:
            raise ValueError(f"{key} is outside the canvas: {rect}")
    keys = ("iq", "cpwc", "ssvd", "power")
    for i, left_key in enumerate(keys):
        a = layout[left_key]
        for right_key in keys[i + 1:]:
            b = layout[right_key]
            overlap = not (a.x + a.width <= b.x or b.x + b.width <= a.x or a.y + a.height <= b.y or b.y + b.height <= a.y)
            if overlap:
                raise ValueError(f"overlap: {left_key} and {right_key}")
