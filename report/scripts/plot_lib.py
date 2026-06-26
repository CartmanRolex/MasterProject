"""Shared PDF bar-chart infrastructure for orange outcome distribution figures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


OUTCOME_RE = re.compile(r"^\s*([0-3])/3 oranges:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)")
MEAN_RE = re.compile(r"^\s*Avg oranges in plate:\s+([\d.]+)/3")
ORDER = [0, 1, 2, 3]
COLORS = {
    0: (0.42, 0.45, 0.50),
    1: (0.93, 0.56, 0.21),
    2: (0.34, 0.63, 0.76),
    3: (0.23, 0.55, 0.31),
}
POLICY_STROKES = {
    "ACT": (0.10, 0.28, 0.66),
    "SmolVLA": (0.58, 0.22, 0.10),
}
# Per-bar recipe/variant label colours (scanning aid; falls back to dark grey).
VARIANT_COLORS = {
    "frozen":   (0.30, 0.30, 0.30),
    "unfrozen": (0.80, 0.45, 0.10),
    "no-tail":  (0.13, 0.45, 0.55),
    "ACT":      (0.10, 0.28, 0.66),
    "SmolVLA":  (0.58, 0.22, 0.10),
}


@dataclass(frozen=True)
class ResultFile:
    label: str
    description: str
    path: Path | None
    dataset: str = ""
    policy: str = ""
    mode: str = ""
    tag: str = ""
    group: str = ""      # family bracket label (e.g. "Teleop\nsubtask"); falls back to dataset
    variant: str = ""    # per-bar label (e.g. "frozen" / "unfrozen" / "no-tail"); falls back to policy
    placeholder: bool = False
    placeholder_lines: tuple[str, ...] = ("future", "data")
    placeholder_value: str = "TBD"


@dataclass(frozen=True)
class ParsedResult:
    outcomes: dict[int, tuple[int, int, float]]
    total: int
    mean: float


def parse_result(result_file: ResultFile) -> ParsedResult:
    if result_file.path is None:
        raise ValueError(f"{result_file.description} has no result path")
    text = result_file.path.read_text(encoding="utf-8")
    outcomes: dict[int, tuple[int, int, float]] = {}
    mean: float | None = None

    for line in text.splitlines():
        outcome_match = OUTCOME_RE.match(line)
        if outcome_match:
            oranges = int(outcome_match.group(1))
            count = int(outcome_match.group(2))
            total = int(outcome_match.group(3))
            pct = 100.0 * count / total
            outcomes[oranges] = (count, total, pct)
            continue

        mean_match = MEAN_RE.match(line)
        if mean_match:
            mean = float(mean_match.group(1))

    missing = [oranges for oranges in ORDER if oranges not in outcomes]
    if missing:
        raise ValueError(f"{result_file.path} is missing outcome rows for: {missing}")
    if mean is None:
        raise ValueError(f"{result_file.path} is missing the mean orange count")

    totals = {outcomes[oranges][1] for oranges in ORDER}
    if len(totals) != 1:
        raise ValueError(f"{result_file.path} has inconsistent episode totals: {sorted(totals)}")

    return ParsedResult(outcomes=outcomes, total=totals.pop(), mean=mean)


def pct_label(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f}%"
    return f"{value:.1f}%"


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def policy_name(result_file: ResultFile) -> str:
    return result_file.policy or result_file.label.splitlines()[0]


def mode_name(result_file: ResultFile) -> str:
    if result_file.mode:
        return result_file.mode
    label_lines = result_file.label.splitlines()
    return label_lines[-1] if label_lines else ""


def dataset_name(result_file: ResultFile) -> str:
    if result_file.dataset:
        return result_file.dataset
    label_lines = result_file.label.splitlines()
    return label_lines[1] if len(label_lines) > 1 else ""


class PdfFigure:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def set_fill(self, rgb: tuple[float, float, float]) -> None:
        self.commands.append(f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg")

    def set_stroke(self, rgb: tuple[float, float, float], width: float = 1.0) -> None:
        self.commands.append(f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} RG")
        self.commands.append(f"{width:.2f} w")

    def rect(self, x: float, y: float, width: float, height: float, fill: bool = True) -> None:
        op = "f" if fill else "S"
        self.commands.append(f"{x:.2f} {y:.2f} {width:.2f} {height:.2f} re {op}")

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 10,
        align: str = "left",
        rgb: tuple[float, float, float] = (0, 0, 0),
        bold: bool = False,
    ) -> None:
        # Helvetica width approximation is sufficient for centering short labels.
        width = len(text) * size * (0.56 if not bold else 0.60)
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        font = "/F2" if bold else "/F1"
        escaped = pdf_escape(text)
        self.commands.append(f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg")
        self.commands.append(f"BT {font} {size:.1f} Tf {x:.2f} {y:.2f} Td ({escaped}) Tj ET")

    def save(self, path: Path) -> None:
        content = "\n".join(self.commands).encode("ascii")
        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.0f} {self.height:.0f}] "
                "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
            ).encode("ascii")
        )
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        objects.append(b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream")

        offsets = [0]
        pdf = bytearray(b"%PDF-1.4\n")
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{index} 0 obj\n".encode("ascii"))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")

        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        pdf.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(pdf)


def draw_badge(
    figure: PdfFigure,
    x: float,
    y: float,
    text: str,
    *,
    width: float = 16,
    height: float = 12,
    fill: tuple[float, float, float] = (0.96, 0.96, 0.94),
    stroke: tuple[float, float, float] = (0.18, 0.18, 0.18),
) -> None:
    figure.set_fill(fill)
    figure.rect(x - width / 2, y, width, height)
    figure.set_stroke(stroke, 0.7)
    figure.rect(x - width / 2, y, width, height, fill=False)
    figure.text(x, y + 3, text, 6.8, "center", rgb=stroke, bold=True)


def draw_figure(
    results: list[tuple[ResultFile, ParsedResult]],
    output_path: Path,
    *,
    bar_w: int = 58,
) -> None:
    figure = PdfFigure(width=720, height=460)
    left, right, bottom, top = 72, 28, 108, 92
    plot_w = figure.width - left - right
    plot_h = figure.height - bottom - top
    x_step = plot_w / len(results)

    figure.text(figure.width / 2, figure.height - 26, "Final orange count per episode", 15, "center", bold=True)
    figure.text(figure.width / 2, figure.height - 43, "Percentage of evaluated episodes", 10, "center", rgb=(0.25, 0.25, 0.25))

    figure.set_stroke((0.78, 0.78, 0.78), 0.6)
    for pct in [0, 25, 50, 75, 100]:
        y = bottom + plot_h * pct / 100
        figure.line(left, y, figure.width - right, y)
        figure.text(left - 10, y - 4, f"{pct}", 8, "right", rgb=(0.25, 0.25, 0.25))

    figure.set_stroke((0.12, 0.12, 0.12), 1.0)
    figure.line(left, bottom, left, bottom + plot_h)
    figure.line(left, bottom, figure.width - right, bottom)

    for index, (result_file, parsed) in enumerate(results):
        x = left + x_step * index + x_step / 2 - bar_w / 2
        y = bottom
        for oranges in ORDER:
            outcomes = parsed.outcomes
            _, _, pct = outcomes[oranges]
            segment_h = plot_h * pct / 100
            figure.set_fill(COLORS[oranges])
            figure.rect(x, y, bar_w, segment_h)
            if segment_h >= 12:
                label_rgb = (1, 1, 1) if oranges in [0, 3] else (0.05, 0.05, 0.05)
                figure.text(x + bar_w / 2, y + segment_h / 2 - 3, pct_label(pct), 7.1, "center", rgb=label_rgb, bold=True)
            y += segment_h

        label_lines = result_file.label.splitlines()
        policy = label_lines[0] if label_lines else ""
        policy_rgb = POLICY_STROKES.get(policy, (0.12, 0.12, 0.12))
        figure.set_stroke(policy_rgb, 1.6)
        figure.rect(x, bottom, bar_w, plot_h, fill=False)

        figure.text(x + bar_w / 2, bottom + plot_h + 22, f"N eval runs={parsed.total}", 7.0, "center", bold=True)
        figure.text(x + bar_w / 2, bottom + plot_h + 10, f"mean={parsed.mean:.2f}/3", 7.6, "center", rgb=(0.2, 0.2, 0.2))

        for line_index, label_line in enumerate(label_lines):
            is_policy = line_index == 0
            figure.text(
                x + bar_w / 2,
                bottom - 19 - 11 * line_index,
                label_line,
                8.4 if is_policy else 7.4,
                "center",
                rgb=policy_rgb if is_policy else (0, 0, 0),
                bold=is_policy,
            )

    legend_y = 20
    legend_x = left + 46
    for oranges in ORDER:
        figure.set_fill(COLORS[oranges])
        figure.rect(legend_x, legend_y, 11, 11)
        figure.text(legend_x + 16, legend_y + 2, f"{oranges}/3 oranges", 8.5)
        legend_x += 122

    figure.save(output_path)


def draw_grouped_figure(
    results: list[tuple[ResultFile, ParsedResult | None]],
    output_path: Path,
    *,
    bar_w: int = 44,
) -> None:
    figure = PdfFigure(width=920, height=512)
    left, right, bottom, top = 74, 34, 138, 96
    plot_w = figure.width - left - right
    plot_h = figure.height - bottom - top
    bar_gap = 11
    group_gap = 46

    # Group by the explicit `group` attribute (fallback: dataset name).
    groups: list[tuple[str, list[tuple[ResultFile, ParsedResult | None]]]] = []
    for result_file, parsed in results:
        key = result_file.group or dataset_name(result_file)
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append((result_file, parsed))

    group_widths = [len(g) * bar_w + (len(g) - 1) * bar_gap for _, g in groups]
    total_w = sum(group_widths) + group_gap * (len(groups) - 1)
    x_cursor = left + (plot_w - total_w) / 2

    figure.text(figure.width / 2, figure.height - 27, "Final orange count per episode", 15, "center", bold=True)
    figure.text(
        figure.width / 2,
        figure.height - 44,
        "Bar height = % of the 100 seeded episodes ending with 0-3 oranges placed; value above each bar is the mean (/3). "
        "Within a family, bars are training recipes.",
        9.0,
        "center",
        rgb=(0.25, 0.25, 0.25),
    )

    figure.set_stroke((0.84, 0.84, 0.82), 0.55)
    for pct in [0, 25, 50, 75, 100]:
        y = bottom + plot_h * pct / 100
        figure.line(left, y, figure.width - right, y)
        figure.text(left - 10, y - 4, f"{pct}", 8, "right", rgb=(0.25, 0.25, 0.25))

    figure.set_stroke((0.12, 0.12, 0.12), 1.0)
    figure.line(left, bottom, left, bottom + plot_h)
    figure.line(left, bottom, figure.width - right, bottom)

    for key, group_results in groups:
        group_w = len(group_results) * bar_w + (len(group_results) - 1) * bar_gap
        group_start = x_cursor
        group_center = group_start + group_w / 2

        for index, (result_file, parsed) in enumerate(group_results):
            x = group_start + index * (bar_w + bar_gap)
            policy = policy_name(result_file)
            policy_rgb = POLICY_STROKES.get(policy, (0.12, 0.12, 0.12))
            variant = result_file.variant or policy
            variant_rgb = VARIANT_COLORS.get(variant, (0.20, 0.20, 0.20))

            if parsed is None or result_file.placeholder:
                figure.set_fill((0.965, 0.965, 0.945))
                figure.rect(x, bottom, bar_w, plot_h)
                figure.set_stroke((0.36, 0.36, 0.34), 1.25)
                figure.rect(x, bottom, bar_w, plot_h, fill=False)
                figure.text(x + bar_w / 2, bottom + plot_h / 2, result_file.placeholder_value, 7.1,
                            "center", rgb=(0.30, 0.30, 0.28), bold=True)
            else:
                y = bottom
                for oranges in ORDER:
                    _, _, pct = parsed.outcomes[oranges]
                    segment_h = plot_h * pct / 100
                    figure.set_fill(COLORS[oranges])
                    figure.rect(x, y, bar_w, segment_h)
                    if segment_h >= 12:
                        label_rgb = (1, 1, 1) if oranges in [0, 3] else (0.05, 0.05, 0.05)
                        figure.text(x + bar_w / 2, y + segment_h / 2 - 3, pct_label(pct), 6.8, "center", rgb=label_rgb, bold=True)
                    y += segment_h
                figure.set_stroke(policy_rgb, 1.5)
                figure.rect(x, bottom, bar_w, plot_h, fill=False)
                figure.text(x + bar_w / 2, bottom + plot_h + 8, f"{parsed.mean:.2f}", 8.4, "center", rgb=(0.15, 0.15, 0.15), bold=True)

            # per-bar recipe/variant label
            figure.text(x + bar_w / 2, bottom - 16, variant, 7.6, "center", rgb=variant_rgb, bold=True)

        # family bracket + (possibly multi-line) label
        figure.set_stroke((0.33, 0.33, 0.33), 0.7)
        figure.line(group_start, bottom - 30, group_start + group_w, bottom - 30)
        figure.line(group_start, bottom - 27, group_start, bottom - 33)
        figure.line(group_start + group_w, bottom - 27, group_start + group_w, bottom - 33)
        for li, gl in enumerate(key.split("\n")):
            figure.text(group_center, bottom - 46 - 11 * li, gl, 8.6, "center", bold=True)

        x_cursor += group_w + group_gap

    # legend: outcome colours + recipe key
    legend_y = 22
    legend_x = left + 26
    for oranges in ORDER:
        figure.set_fill(COLORS[oranges])
        figure.rect(legend_x, legend_y, 10, 10)
        figure.text(legend_x + 14, legend_y + 2, f"{oranges}/3", 8.0)
        legend_x += 58
    legend_x += 18
    figure.text(legend_x, legend_y + 2, "recipe:", 8.0, rgb=(0.2, 0.2, 0.2))
    legend_x += 44
    for variant in ("frozen", "unfrozen", "no-tail"):
        figure.text(legend_x, legend_y + 2, variant, 8.0, rgb=VARIANT_COLORS[variant], bold=True)
        legend_x += len(variant) * 8.0 * 0.56 + 16

    figure.save(output_path)
