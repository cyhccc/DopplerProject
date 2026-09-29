from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon
import numpy as np

from .figure_spec import FIGURE_HEIGHT_MM, FIGURE_WIDTH_MM, PNG_DPI, RectMM, build_layout

PALA_NX, PALA_NZ = 256, 180
PALA_DB_MIN, PALA_DB_MAX = -40.0, 0.0
PANEL_WIDTH_MM, PANEL_HEIGHT_MM = 38.0, 28.0
PANEL_DPI = 600
PALA_DX_MM = 0.50730519480519476 * 0.09856
SCALE_BAR_PIXELS = round(1.0 / PALA_DX_MM)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimSun", "Times New Roman", "DejaVu Sans"],
    "font.size": 7.5, "svg.fonttype": "none", "axes.unicode_minus": False,
})

FLOW, BLUE, ORANGE, PALE = "#344F66", "#2F73B7", "#D97A35", "#F7FAFC"


def _card(ax, rect: RectMM, title: str, linewidth: float = 0.8) -> None:
    ax.add_patch(FancyBboxPatch((rect.x, rect.y), rect.width, rect.height,
        boxstyle="round,pad=0.5,rounding_size=1.3", facecolor="white",
        edgecolor="#8EA4B5", linewidth=linewidth, zorder=1))
    title_size = 6.2 if len(title) > 10 else 7.2
    ax.text(rect.x + rect.width / 2, rect.y + rect.height - 3.2, title,
            ha="center", va="center", weight="bold", fontsize=title_size, color="#173C5D")


def draw_acquisition(ax, r: RectMM) -> None:
    ax.text(r.x+r.width/2, r.y+r.height+3, "超声平面波采集", ha="center", weight="bold", color="#173C5D")
    ax.add_patch(Ellipse((r.x+9, r.y+13), 13, 18, facecolor="#F5E2D8", edgecolor="#9E6C5B", lw=.7))
    ax.add_patch(Ellipse((r.x+5.3, r.y+19), 4, 5, facecolor="#F5E2D8", edgecolor="#9E6C5B", lw=.6))
    ax.add_patch(Ellipse((r.x+12.7, r.y+19), 4, 5, facecolor="#F5E2D8", edgecolor="#9E6C5B", lw=.6))
    ax.add_patch(FancyBboxPatch((r.x+3, r.y+32), 12, 3, boxstyle="round,pad=.2", fc="#415C70", ec="none"))
    for x in np.linspace(r.x+3.6, r.x+14.4, 11): ax.plot([x,x], [r.y+32.3,r.y+34.7], color="white", lw=.35)
    for dx in (-3.2, 0, 3.2):
        ax.plot([r.x+9+dx, r.x+9+dx*0.35], [r.y+31.5, r.y+22], color=BLUE, lw=.7, alpha=.75)
    ax.text(r.x+9, r.y+4, "多角度连续帧RF数据", ha="center", fontsize=5.5, color=FLOW)


def draw_iq_demodulation(ax, r: RectMM) -> None:
    _card(ax, r, "IQ解调")
    x=np.linspace(r.x+2,r.x+r.width-2,120); t=np.linspace(0,7*np.pi,120)
    ax.plot(x, r.y+25+2*np.sin(t), color="#606A72", lw=.7)
    ax.annotate("", (r.x+r.width/2,r.y+18), (r.x+r.width/2,r.y+22), arrowprops=dict(arrowstyle="-|>",lw=.7,color=FLOW))
    ax.plot(x, r.y+13+1.6*np.sin(t/3), color=BLUE, lw=1)
    ax.plot(x, r.y+8+1.6*np.cos(t/3), color=ORANGE, lw=1)
    ax.text(r.x+3.2,r.y+13,"I",color=BLUE,ha="center",va="center",fontsize=7,weight="bold",
            bbox=dict(boxstyle="round,pad=.12",fc="white",ec=BLUE,lw=.55))
    ax.text(r.x+3.2,r.y+8,"Q",color=ORANGE,ha="center",va="center",fontsize=7,weight="bold",
            bbox=dict(boxstyle="round,pad=.12",fc="white",ec=ORANGE,lw=.55))
    ax.text(r.x+r.width/2,r.y+3,"正交下变频 · 低通滤波",ha="center",fontsize=5.5,color=FLOW)


def _cpwc_wavefront_segments(r: RectMM) -> tuple[tuple[float, float, float, float, str], ...]:
    """Return three visibly distinct CPWC plane-wave fronts."""
    x0, x1 = r.x + 4, r.x + r.width - 4
    center_y = r.y + 26
    half_rise = 1.3
    return (
        (x0, center_y + half_rise, x1, center_y - half_rise, "-θ"),
        (x0, center_y, x1, center_y, "0°"),
        (x0, center_y - half_rise, x1, center_y + half_rise, "+θ"),
    )


def _cpwc_wavefront_label_positions(r: RectMM) -> tuple[tuple[float, float, str], ...]:
    """Place angle labels beside, not above, the title-safe wavefront area."""
    return tuple(
        (r.x + 2.2, wave_y0, label)
        for _, wave_y0, _, _, label in _cpwc_wavefront_segments(r)
    )


def draw_cpwc_das(ax, r: RectMM) -> None:
    _card(ax, r, "CPWC-DAS波束形成", 1.3)
    y0=r.y+24; ax.plot([r.x+4,r.x+r.width-4],[y0,y0],color="#243F55",lw=2.2)
    for x in np.linspace(r.x+4,r.x+r.width-4,9): ax.plot([x,x],[y0-.8,y0+.8],color="white",lw=.5)
    for x0, wave_y0, x1, wave_y1, _ in _cpwc_wavefront_segments(r):
        ax.plot([x0, x1], [wave_y0, wave_y1], color=BLUE, lw=.75)
        mid_x, mid_y = (x0 + x1) / 2, (wave_y0 + wave_y1) / 2
        ax.plot([mid_x, r.x+r.width/2], [mid_y, y0], color=BLUE, lw=.55)
    for label_x, label_y, label in _cpwc_wavefront_label_positions(r):
        ax.text(label_x, label_y, label, ha="center", va="center", fontsize=4.8, color=BLUE)
    px,py=r.x+r.width/2,r.y+11
    ax.add_patch(Circle((px,py),.8,fc=ORANGE,ec="white",lw=.4,zorder=4)); ax.text(px+1.2,py,"r",va="center",fontsize=6,style="italic")
    for x in np.linspace(r.x+5,r.x+r.width-5,7): ax.plot([x,px],[y0,py],ls=(0,(2,2)),color="#7B8790",lw=.45)
    ax.text(px,r.y+4,"多角度延时求和与相干复合",ha="center",fontsize=5.4,color=FLOW)


def draw_ssvd_ssm(ax, r: RectMM) -> None:
    _card(ax, r, "sSVD+SSM杂波滤波")
    ax.text(r.x+5,r.y+27.5,"X",ha="center",va="center",fontsize=5.2,
            bbox=dict(boxstyle="round,pad=.2",fc="#EEF3F7",ec="#9DAFBD",lw=.45))
    ax.annotate("",(r.x+10,r.y+27.5),(r.x+6.2,r.y+27.5),arrowprops=dict(arrowstyle="->",lw=.6,color=FLOW))
    ax.text(r.x+12,r.y+27.5,"X  X",ha="center",va="center",fontsize=4.5,
            bbox=dict(boxstyle="round,pad=.2",fc="#EEF3F7",ec="#9DAFBD",lw=.45))
    ax.text(r.x+12,r.y+28.8,"H",ha="center",va="center",fontsize=3.6,color=FLOW)
    ax.annotate("",(r.x+9,r.y+23),(r.x+12,r.y+26.2),arrowprops=dict(arrowstyle="->",lw=.6,color=FLOW))
    ax.text(r.x+7,r.y+21.5,"特征分解",ha="center",va="center",fontsize=4.4,
            bbox=dict(boxstyle="round,pad=.2",fc="#EEF3F7",ec="#9DAFBD",lw=.45))
    ax.annotate("",(r.x+14,r.y+21.5),(r.x+10.2,r.y+21.5),arrowprops=dict(arrowstyle="->",lw=.6,color=FLOW))
    ax.text(r.x+18,r.y+21.5,"U=XV/σ",ha="center",va="center",fontsize=4.4,
            bbox=dict(boxstyle="round,pad=.2",fc="#EEF3F7",ec="#9DAFBD",lw=.45))
    u_to_ssm = ax.annotate(
        "", (r.x+11.5, r.y+18.4), (r.x+17.3, r.y+20.3),
        arrowprops=dict(arrowstyle="-|>", lw=.75, color=FLOW, mutation_scale=7),
    )
    u_to_ssm.arrow_patch.set_gid("ssvd-u-to-ssm")
    heat=np.array([[.1,.3,.2,.1],[.2,.8,.5,.2],[.1,.5,1,.3],[.05,.2,.3,.7]])
    ax.imshow(heat,extent=(r.x+6,r.x+12,r.y+8,r.y+16),origin="lower",cmap="Blues",vmin=0,vmax=1,zorder=2,aspect="auto")
    ax.text(r.x+9,r.y+18,"SSM",ha="center",fontsize=5.5,weight="bold")
    ax.annotate("",(r.x+17,r.y+12),(r.x+12.5,r.y+12),arrowprops=dict(arrowstyle="->",lw=.6,color=FLOW))
    ax.text(r.x+18,r.y+12,"β",ha="center",va="center",fontsize=10,color=ORANGE,weight="bold")
    ax.text(r.x+r.width/2,r.y+3.2,"奇异谱软掩膜抑制组织杂波",ha="center",fontsize=5.2,color=FLOW)


def draw_power_sum(ax, r: RectMM) -> None:
    _card(ax, r, "功率累积")
    rng=np.random.default_rng(3)
    for i in range(3):
        data=np.exp(-((np.indices((12,14))[0]-6-i*.5)**2+(np.indices((12,14))[1]-7)**2)/12)+rng.random((12,14))*.12
        ax.imshow(data,extent=(r.x+2+i*4.2,r.x+6+i*4.2,r.y+19,r.y+26),cmap="magma",origin="lower",aspect="auto",zorder=2)
    ax.text(r.x+r.width/2,r.y+14,
            r"$P(n)=\sum_f|\mathrm{IQ}_{\mathrm{filtered}}(n,f)|^2$",
            ha="center",fontsize=4.8,color="#173C5D")
    thumb=np.outer(np.hanning(15),np.hanning(20)); ax.imshow(thumb,extent=(r.x+5,r.x+r.width-5,r.y+3,r.y+11),cmap="hot",origin="lower",aspect="auto",zorder=2)


def draw_gpu_bus(ax, layout) -> None:
    g=layout["gpu"]; cx=g.x+g.width/2; chip_w,chip_h=13,6.5
    chip=FancyBboxPatch((cx-chip_w/2,g.y+1.2),chip_w,chip_h,
        boxstyle="round,pad=.25,rounding_size=.8",fc="#EAF4FF",ec=BLUE,lw=1.0)
    chip.set_gid("gpu-chip"); ax.add_patch(chip)
    for dx in np.linspace(-4.5,4.5,4):
        ax.plot([cx+dx,cx+dx],[g.y+2.2,g.y+6.7],color="#8EB9DF",lw=.45)
    for y in np.linspace(g.y+2.5,g.y+6.5,3):
        ax.plot([cx-5.2,cx+5.2],[y,y],color="#8EB9DF",lw=.45)
    for dx in np.linspace(-5,5,5):
        ax.plot([cx+dx,cx+dx],[g.y+.4,g.y+1.2],color=BLUE,lw=.6)
        ax.plot([cx+dx,cx+dx],[g.y+7.7,g.y+8.5],color=BLUE,lw=.6)
    ax.text(cx,g.y+4.5,"GPU端执行",ha="center",va="center",color="#125FA8",weight="bold",fontsize=6.2,
            bbox=dict(fc="#EAF4FF",ec="none",pad=.3))
    bus_y=g.y+g.height-.2; ax.plot([33,108],[bus_y,bus_y],color=BLUE,lw=.8,ls=(0,(3,2)))
    ax.plot([cx,cx],[g.y+7.7,bus_y],color=BLUE,lw=.8,ls=(0,(3,2)))
    for key in ("iq","cpwc","ssvd","power"):
        r=layout[key]
        link, = ax.plot(
            [r.x+r.width/2, r.x+r.width/2], [bus_y, r.y],
            color=BLUE, lw=.8, ls=(0,(3,2)), solid_capstyle="round",
        )
        link.set_gid(f"gpu-link-{key}")


def _flow_arrow(ax, left: RectMM, right: RectMM) -> None:
    y=(left.y+left.height/2+right.y+right.height/2)/2
    ax.annotate("",(right.x-.8,y),(left.x+left.width+.8,y),arrowprops=dict(arrowstyle="-|>",color=FLOW,lw=1.1))


def load_pala_power_db(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = PALA_NX * PALA_NZ
    if values.size != expected:
        raise ValueError(f"Expected {expected} float32 values, got {values.size}: {path}")
    power = values.reshape((PALA_NZ, PALA_NX), order="C")
    peak = float(np.nanmax(power))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError(f"Invalid PALA power peak {peak}: {path}")
    return 10.0 * np.log10(
        np.maximum(power, np.finfo(np.float32).tiny) / peak
    )


def render_compact_pala_panel(power_db: np.ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(
        figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4),
        dpi=PANEL_DPI,
        facecolor="white",
    )
    image_axis = figure.add_axes((0.035, 0.055, 0.79, 0.91))
    image = image_axis.imshow(
        power_db,
        cmap="hot",
        vmin=PALA_DB_MIN,
        vmax=PALA_DB_MAX,
        origin="upper",
        aspect="auto",
    )
    image_axis.set_axis_off()

    scale_y = PALA_NZ - 13
    scale_x0, scale_x1 = 12, 12 + SCALE_BAR_PIXELS
    image_axis.plot(
        [scale_x0, scale_x1], [scale_y, scale_y],
        color="white", linewidth=1.5, solid_capstyle="butt",
    )
    image_axis.text(
        (scale_x0 + scale_x1) / 2, scale_y - 5, "1 mm",
        color="white", fontsize=4.6, ha="center", va="bottom",
    )

    colorbar_axis = figure.add_axes((0.86, 0.20, 0.035, 0.60))
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=(-40, -20, 0))
    colorbar.ax.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=1)
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.set_title("dB", fontsize=4.5, pad=2)

    figure.savefig(output_path, dpi=PANEL_DPI, facecolor="white")
    plt.close(figure)
    return output_path


def render_pipeline(power_sum_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    layout = build_layout(); power_db = load_pala_power_db(Path(power_sum_path))
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM/25.4, FIGURE_HEIGHT_MM/25.4), dpi=PNG_DPI, facecolor="white")
    ax = fig.add_axes((0,0,1,1)); ax.set_xlim(0,FIGURE_WIDTH_MM); ax.set_ylim(0,FIGURE_HEIGHT_MM); ax.axis("off")
    draw_acquisition(ax, layout["acquisition"])
    draw_iq_demodulation(ax, layout["iq"]); draw_cpwc_das(ax, layout["cpwc"])
    draw_ssvd_ssm(ax, layout["ssvd"]); draw_power_sum(ax, layout["power"])
    for a,b in zip(("acquisition","iq","cpwc","ssvd","power"),("iq","cpwc","ssvd","power","output")):
        _flow_arrow(ax,layout[a],layout[b])
    out=layout["output"]
    ax.text(out.x+out.width/2,out.y+out.height+4,"PALA大鼠脑功率多普勒\n（800帧）",
            ha="center",va="center",weight="bold",fontsize=5.9,linespacing=1.0,color="#173C5D")
    image=ax.imshow(power_db,extent=(out.x,out.x+out.width-4,out.y,out.y+out.height),origin="upper",cmap="hot",vmin=-40,vmax=0,aspect="auto",zorder=1)
    x0=out.x+2; bar_mm=(out.width-4)*SCALE_BAR_PIXELS/PALA_NX
    ax.plot([x0,x0+bar_mm],[out.y+3,out.y+3],color="white",lw=1.4); ax.text(x0+bar_mm/2,out.y+4.4,"1 mm",color="white",ha="center",fontsize=5)
    cax=fig.add_axes(((out.x+out.width-4.2)/FIGURE_WIDTH_MM,out.y/FIGURE_HEIGHT_MM,2.0/FIGURE_WIDTH_MM,out.height/FIGURE_HEIGHT_MM))
    cb=fig.colorbar(image,cax=cax,ticks=(-40,-20,0)); cb.ax.tick_params(labelsize=5,length=2,pad=1); cb.outline.set_linewidth(.4); cb.ax.set_title("dB",fontsize=5,pad=2)
    draw_gpu_bus(ax,layout)
    svg_path=output_dir/"cf_ssvd_gpu_pipeline_150mm.svg"; png_path=output_dir/"cf_ssvd_gpu_pipeline_150mm_600dpi.png"
    fig.savefig(svg_path,facecolor="white"); fig.savefig(png_path,dpi=PNG_DPI,facecolor="white")
    plt.close(fig); return svg_path,png_path


def main() -> None:
    parser=argparse.ArgumentParser(description="Render the CF-sSVD-GPU algorithm pipeline")
    parser.add_argument("--power-sum",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(); render_pipeline(args.power_sum,args.output_dir)


if __name__ == "__main__":
    main()
