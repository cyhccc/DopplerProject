import unittest
import xml.etree.ElementTree as ET
import re
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from lxml import etree as LET
from PIL import Image

from paper_figures.cf_ssvd_gpu_pipeline.figure_spec import (
    FIGURE_HEIGHT_MM, FIGURE_WIDTH_MM, FORBIDDEN_FIG1_TERMS,
    MODULES, build_layout, validate_layout,
)
from paper_figures.cf_ssvd_gpu_pipeline.render_cf_ssvd_gpu_pipeline import (
    PALA_DX_MM, SCALE_BAR_PIXELS, _cpwc_wavefront_label_positions,
    _cpwc_wavefront_segments,
    load_pala_power_db, render_compact_pala_panel,
    render_pipeline,
)
from paper_figures.cf_ssvd_gpu_pipeline.replace_figure1_in_docx import replace_figure1

PALA_POWER_SUM = Path(r"D:\DopplerProject\CudaMatlabCpuQuality\outputs\power_doppler_7883227_rf001_to_001_cuda_covsvd_cutoff50_800_red_current_power_sum.bin")
SOURCE_DOCX = Path(r"C:\Users\11141\Desktop\7.71.docx")
FIGURE_PNG = Path(__file__).with_name("cf_ssvd_gpu_pipeline_150mm_600dpi.png")
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"w": W_NS, "a": A_NS, "wp": WP_NS, "r": R_NS}


def _package_facts(path: Path) -> dict[str, object]:
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        xml = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return {
            "drawings": len(xml.findall(".//w:drawing", ns)),
            "comments": sorted(name for name in names if "comment" in name.lower()),
            "insertions": len(xml.findall(".//w:ins", ns)),
            "deletions": len(xml.findall(".//w:del", ns)),
        }


def _write_review_fixture(source: Path, output: Path) -> None:
    with ZipFile(source) as archive:
        entries = {info.filename: (info, archive.read(info.filename)) for info in archive.infolist()}
    root = LET.fromstring(entries["word/document.xml"][1])
    narrative = next(
        p for p in root.xpath(".//w:body/w:p", namespaces=NS)
        if "图1为CF-sSVD-GPU方法的整体结构" in "".join(p.xpath(".//w:t/text()", namespaces=NS))
    )
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)
    figure = next(p for p in paragraphs[paragraphs.index(narrative) + 1:] if p.xpath(".//w:drawing", namespaces=NS))
    drawing_run = figure.xpath(".//w:r[.//w:drawing]", namespaces=NS)[0]
    parent = drawing_run.getparent()
    position = parent.index(drawing_run)
    parent.remove(drawing_run)
    insertion = LET.Element(f"{{{W_NS}}}ins")
    insertion.set(f"{{{W_NS}}}id", "901")
    insertion.set(f"{{{W_NS}}}author", "Review Fixture")
    insertion.append(drawing_run)
    parent.insert(position, LET.Element(f"{{{W_NS}}}commentRangeStart", {f"{{{W_NS}}}id": "77"}))
    parent.insert(position + 1, insertion)
    parent.insert(position + 2, LET.Element(f"{{{W_NS}}}commentRangeEnd", {f"{{{W_NS}}}id": "77"}))
    reference_run = LET.Element(f"{{{W_NS}}}r")
    LET.SubElement(reference_run, f"{{{W_NS}}}commentReference", {f"{{{W_NS}}}id": "77"})
    parent.insert(position + 3, reference_run)
    deletion = LET.Element(f"{{{W_NS}}}del")
    deletion.set(f"{{{W_NS}}}id", "902")
    deletion.set(f"{{{W_NS}}}author", "Review Fixture")
    deleted_run = LET.SubElement(deletion, f"{{{W_NS}}}r")
    deleted_text = LET.SubElement(deleted_run, f"{{{W_NS}}}delText")
    deleted_text.text = "审阅夹具删除文本"
    parent.insert(position + 4, deletion)
    entries["word/document.xml"] = (entries["word/document.xml"][0], LET.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True))
    with ZipFile(output, "w", ZIP_DEFLATED) as target:
        for _, (info, data) in entries.items():
            target.writestr(info, data)


def _figure_media_facts(path: Path) -> dict[str, object]:
    with ZipFile(path) as archive:
        root = LET.fromstring(archive.read("word/document.xml"))
        rels = LET.fromstring(archive.read("word/_rels/document.xml.rels"))
        relmap = {rel.get("Id"): rel.get("Target") for rel in rels}
        paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)
        narrative = next(p for p in paragraphs if "图1为CF-sSVD-GPU方法的整体结构" in "".join(p.xpath(".//w:t/text()", namespaces=NS)))
        figure = next(p for p in paragraphs[paragraphs.index(narrative) + 1:] if p.xpath(".//w:drawing", namespaces=NS))
        rids = [b.get(f"{{{R_NS}}}embed") for b in root.xpath(".//a:blip", namespaces=NS)]
        targets = [relmap[rid] for rid in rids]
        hashes = [hashlib.sha256(archive.read("word/" + target)).hexdigest() for target in targets]
        extent = figure.xpath(".//wp:extent", namespaces=NS)[0]
        media_names = sorted(name for name in archive.namelist() if name.startswith("word/media/"))
        referenced = {"word/" + target for target in relmap.values() if target and target.startswith("media/")}
        return {
            "root": root, "figure": figure, "rids": rids, "targets": targets, "hashes": hashes,
            "cx": int(extent.get("cx")), "cy": int(extent.get("cy")),
            "media_names": media_names, "referenced_media": referenced,
            "image_relationships": sum(1 for rel in rels if (rel.get("Type") or "").endswith("/image")),
        }


class FigureSpecTests(unittest.TestCase):
    def test_forbidden_terms_cover_figure3_thread_block_structure(self) -> None:
        expected_terms = {
            "Thread0", "Thread31", "warp", "frame lane", "shared memory",
            "线程块结构", "thread block",
        }
        self.assertTrue(expected_terms.issubset(set(FORBIDDEN_FIG1_TERMS)))

    def test_algorithm_chain_is_exact(self) -> None:
        self.assertEqual(
            [module.label for module in MODULES],
            ["IQ解调", "CPWC-DAS波束形成", "sSVD+SSM杂波滤波", "功率累积"],
        )
        self.assertEqual(MODULES[1].visual_kind, "cpwc_das")

    def test_figure1_does_not_duplicate_figure3(self) -> None:
        all_text = " ".join(module.label + " " + module.visual_kind for module in MODULES)
        for forbidden in FORBIDDEN_FIG1_TERMS:
            self.assertNotIn(forbidden.lower(), all_text.lower())

    def test_layout_is_valid(self) -> None:
        layout = build_layout()
        validate_layout(layout)
        for rect in layout.values():
            self.assertGreaterEqual(rect.x, 0)
            self.assertGreaterEqual(rect.y, 0)
            self.assertLessEqual(rect.x + rect.width, FIGURE_WIDTH_MM)
            self.assertLessEqual(rect.y + rect.height, FIGURE_HEIGHT_MM)


class PalaPanelTests(unittest.TestCase):
    def test_load_real_pala_power_uses_cuda_c_row_major_layout(self) -> None:
        raw = np.fromfile(PALA_POWER_SUM, dtype=np.float32)
        power_c = raw.reshape((180, 256), order="C")
        expected_db = 10.0 * np.log10(
            np.maximum(power_c, np.finfo(np.float32).tiny) / float(np.max(power_c))
        )

        np.testing.assert_allclose(
            load_pala_power_db(PALA_POWER_SUM), expected_db, rtol=1e-6, atol=1e-6
        )

    def test_one_mm_scale_bar_matches_pala_lateral_sampling(self) -> None:
        self.assertAlmostEqual(SCALE_BAR_PIXELS * PALA_DX_MM, 1.0, places=2)

    def test_load_real_pala_power(self) -> None:
        image_db = load_pala_power_db(PALA_POWER_SUM)
        self.assertEqual(image_db.shape, (180, 256))
        self.assertAlmostEqual(float(np.max(image_db)), 0.0, places=5)
        self.assertTrue(np.isfinite(image_db).all())

    def test_compact_panel_is_png(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "pala.png"
            render_compact_pala_panel(load_pala_power_db(PALA_POWER_SUM), output)
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 850)
                self.assertGreaterEqual(image.height, 600)


class PipelineRenderTests(unittest.TestCase):
    def test_exports_required_semantics(self) -> None:
        with TemporaryDirectory() as tmp:
            svg_path, png_path = render_pipeline(PALA_POWER_SUM, Path(tmp))
            svg_text = svg_path.read_text(encoding="utf-8")
            for required in (
                "IQ解调", "CPWC-DAS波束形成", "sSVD+SSM杂波滤波", "功率累积", "GPU端执行",
            ):
                self.assertIn(required, svg_text)
            for forbidden in FORBIDDEN_FIG1_TERMS:
                self.assertNotIn(forbidden.lower(), svg_text.lower())
            ET.parse(svg_path)
            with Image.open(png_path) as image:
                self.assertLessEqual(abs(image.width - 3543), 1)
                self.assertLessEqual(abs(image.height - 1559), 1)

    def test_input_semantics_are_rf_before_iq_demodulation(self) -> None:
        with TemporaryDirectory() as tmp:
            svg_path, _ = render_pipeline(PALA_POWER_SUM, Path(tmp))
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("多角度连续帧RF数据", svg_text)
            self.assertNotIn("800帧复数IQ", svg_text)

    def test_gpu_chip_is_below_cards_with_four_dashed_platform_links(self) -> None:
        layout = build_layout()
        algorithm_cards = [layout[key] for key in ("iq", "cpwc", "ssvd", "power")]
        self.assertLessEqual(
            layout["gpu"].y + layout["gpu"].height,
            min(card.y for card in algorithm_cards),
        )
        with TemporaryDirectory() as tmp:
            svg_path, _ = render_pipeline(PALA_POWER_SUM, Path(tmp))
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn('id="gpu-chip"', svg_text)
            self.assertEqual(len(re.findall(r'id="gpu-link-[^"]+"', svg_text)), 4)
            root = ET.parse(svg_path).getroot()
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            for key in ("iq", "cpwc", "ssvd", "power"):
                group = root.find(f".//svg:g[@id='gpu-link-{key}']", namespace)
                self.assertIsNotNone(group)
                paths = group.findall("svg:path", namespace)
                self.assertEqual(len(paths), 1, f"gpu-link-{key} must be one unheaded platform link")
                self.assertTrue(
                    all("stroke-dasharray" in path.attrib.get("style", "") for path in paths),
                    f"gpu-link-{key} must remain visually dashed",
                )
                self.assertNotIn("marker-end", paths[0].attrib)
                self.assertIn("fill: none", paths[0].attrib.get("style", ""))

    def test_cpwc_plane_wavefronts_have_three_distinct_slopes(self) -> None:
        segments = _cpwc_wavefront_segments(build_layout()["cpwc"])
        self.assertEqual(len(segments), 3)
        slopes = [round((y1 - y0) / (x1 - x0), 3) for x0, y0, x1, y1, _ in segments]
        self.assertEqual(len(set(slopes)), 3)
        self.assertLess(slopes[0], slopes[1])
        self.assertLess(slopes[1], slopes[2])

    def test_cpwc_wavefronts_and_labels_stay_below_title_safe_area(self) -> None:
        card = build_layout()["cpwc"]
        title_safe_bottom = card.y + card.height - 5.5
        wavefront_top = max(max(y0, y1) for _, y0, _, y1, _ in _cpwc_wavefront_segments(card))
        label_top = max(y + 0.8 for _, y, _ in _cpwc_wavefront_label_positions(card))
        self.assertLess(wavefront_top, title_safe_bottom)
        self.assertLess(label_top, title_safe_bottom)

    def test_ssvd_u_has_explicit_arrow_to_ssm(self) -> None:
        with TemporaryDirectory() as tmp:
            svg_path, _ = render_pipeline(PALA_POWER_SUM, Path(tmp))
            root = ET.parse(svg_path).getroot()
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            group = root.find(".//svg:g[@id='ssvd-u-to-ssm']", namespace)
            self.assertIsNotNone(group)
            self.assertGreaterEqual(len(group.findall("svg:path", namespace)), 2)


class OutputQualityTests(unittest.TestCase):
    def test_png_has_white_background_and_safe_margin(self) -> None:
        with TemporaryDirectory() as tmp:
            _, png_path = render_pipeline(PALA_POWER_SUM, Path(tmp))
            pixels = np.asarray(Image.open(png_path).convert("RGB"))
        self.assertTrue(np.all(pixels[0, 0] >= 248))
        nonwhite = np.any(pixels < 245, axis=2)
        ys, xs = np.where(nonwhite)
        self.assertGreater(xs.min(), 5)
        self.assertLess(xs.max(), pixels.shape[1] - 6)
        self.assertGreater(ys.min(), 5)
        self.assertLess(ys.max(), pixels.shape[0] - 6)


class DocxReplacementTests(unittest.TestCase):
    def test_preserves_review_nodes_and_drawing_insertion_ancestor(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "review.docx"
            output = Path(tmp) / "replaced.docx"
            _write_review_fixture(SOURCE_DOCX, fixture)
            before_facts = _figure_media_facts(fixture)
            before_deletion = LET.tostring(
                before_facts["figure"].xpath("./w:del[@w:id='902']", namespaces=NS)[0]
            )

            replace_figure1(fixture, FIGURE_PNG, output)

            facts = _figure_media_facts(output)
            figure = facts["figure"]
            self.assertEqual(len(figure.xpath("./w:commentRangeStart[@w:id='77']", namespaces=NS)), 1)
            self.assertEqual(len(figure.xpath("./w:commentRangeEnd[@w:id='77']", namespaces=NS)), 1)
            self.assertEqual(len(figure.xpath("./w:r/w:commentReference[@w:id='77']", namespaces=NS)), 1)
            self.assertEqual(len(figure.xpath("./w:ins[@w:id='901']/w:r/w:drawing", namespaces=NS)), 1)
            deleted = figure.xpath("./w:del[@w:id='902']/w:r/w:delText", namespaces=NS)
            self.assertEqual(len(deleted), 1)
            self.assertEqual(deleted[0].text, "审阅夹具删除文本")
            self.assertEqual(
                before_deletion,
                LET.tostring(figure.xpath("./w:del[@w:id='902']", namespaces=NS)[0]),
            )

    def test_fails_closed_when_figure1_anchors_are_absent(self) -> None:
        with TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "no-anchor.docx"
            output = Path(tmp) / "replaced.docx"
            with ZipFile(SOURCE_DOCX) as archive:
                entries = {info.filename: (info, archive.read(info.filename)) for info in archive.infolist()}
            root = LET.fromstring(entries["word/document.xml"][1])
            for paragraph in root.xpath(".//w:p", namespaces=NS):
                nodes = paragraph.xpath(".//w:t", namespaces=NS)
                if "图1为CF-sSVD-GPU方法的整体结构" in "".join(node.text or "" for node in nodes):
                    nodes[0].text = "无图号锚点"
                    for node in nodes[1:]:
                        node.text = ""
            entries["word/document.xml"] = (
                entries["word/document.xml"][0],
                LET.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True),
            )
            with ZipFile(fixture, "w", ZIP_DEFLATED) as target:
                for _, (info, data) in entries.items():
                    target.writestr(info, data)
            with self.assertRaisesRegex(ValueError, "图1定位锚点"):
                replace_figure1(fixture, FIGURE_PNG, output)

    def test_replaces_only_figure1_and_preserves_review_markup(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "replaced.docx"
            source_hash = hashlib.sha256(SOURCE_DOCX.read_bytes()).hexdigest()
            before = _package_facts(SOURCE_DOCX)
            before_media = _figure_media_facts(SOURCE_DOCX)

            replace_figure1(SOURCE_DOCX, FIGURE_PNG, output)

            self.assertTrue(output.exists())
            self.assertEqual(source_hash, hashlib.sha256(SOURCE_DOCX.read_bytes()).hexdigest())
            self.assertEqual(before, _package_facts(output))
            after_media = _figure_media_facts(output)
            self.assertEqual(before_media["rids"][0], after_media["rids"][0])
            self.assertNotEqual(before_media["targets"][0], after_media["targets"][0])
            self.assertNotEqual(before_media["hashes"][0], after_media["hashes"][0])
            self.assertEqual(before_media["targets"][1:], after_media["targets"][1:])
            self.assertEqual(before_media["hashes"][1:], after_media["hashes"][1:])
            self.assertEqual(before_media["image_relationships"], after_media["image_relationships"])
            self.assertEqual(set(after_media["media_names"]), after_media["referenced_media"])
            self.assertEqual(after_media["cx"], 150 * 36000)
            with Image.open(FIGURE_PNG) as image:
                expected_cy = round(after_media["cx"] * image.height / image.width)
            self.assertEqual(after_media["cy"], expected_cy)
            text = "".join(after_media["root"].itertext())
            self.assertEqual(text.count("图1 CF-sSVD-GPU超快功率多普勒成像处理流程图"), 1)
            self.assertEqual(text.count("Fig. 1 CF-sSVD-GPU ultrafast power Doppler imaging pipeline"), 1)
            self.assertNotIn("图1 功率多普勒成像处理流程图", text)
