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


@dataclass(frozen=True)
class ResultFile:
    label: str
    description: str
    path: Path


@dataclass(frozen=True)
class ParsedResult:
    outcomes: dict[int, tuple[int, int, float]]
    total: int
    mean: float


def parse_result(result_file: ResultFile) -> ParsedResult:
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
    figure.text(24, bottom + plot_h / 2, "% episodes", 9, "center", rgb=(0.2, 0.2, 0.2))

    for index, (result_file, parsed) in enumerate(results):
        x = left + x_step * index + x_step / 2 - bar_w / 2
        y = bottom
        for oranges in ORDER:
            outcomes = parsed.outcomes
            _, _, pct = outcomes[oranges]
            segment_h = plot_h * pct / 100
            figure.set_fill(COLORS[oranges])
            figure.rect(x, y, bar_w, segment_h)
            label_rgb = (1, 1, 1) if oranges in [0, 3] else (0.05, 0.05, 0.05)
            figure.text(x + bar_w / 2, y + segment_h / 2 - 3, pct_label(pct), 7.1, "center", rgb=label_rgb, bold=True)
            y += segment_h

        figure.text(x + bar_w / 2, bottom + plot_h + 22, f"N={parsed.total}", 8.3, "center", bold=True)
        figure.text(x + bar_w / 2, bottom + plot_h + 10, f"mean={parsed.mean:.2f}/3", 7.6, "center", rgb=(0.2, 0.2, 0.2))

        for line_index, label_line in enumerate(result_file.label.splitlines()):
            figure.text(x + bar_w / 2, bottom - 19 - 11 * line_index, label_line, 7.8, "center")

    legend_y = 20
    legend_x = left + 46
    for oranges in ORDER:
        figure.set_fill(COLORS[oranges])
        figure.rect(legend_x, legend_y, 11, 11)
        figure.text(legend_x + 16, legend_y + 2, f"{oranges}/3 oranges", 8.5)
        legend_x += 122

    figure.save(output_path)
