"""Replace manuscript Figure 1 with a minimal, review-markup-safe OOXML patch."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from lxml import etree
from PIL import Image


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"w": W, "a": A, "wp": WP, "r": R}
EMU_PER_MM = 36000
FIGURE_WIDTH_MM = 150.0
ZH_CAPTION = "图1 CF-sSVD-GPU超快功率多普勒成像处理流程图"
EN_CAPTION = "Fig. 1 CF-sSVD-GPU ultrafast power Doppler imaging pipeline"


def _paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text() | .//w:delText/text()", namespaces=NS))


def _find_figure1_paragraph(root: etree._Element) -> tuple[etree._Element, list[etree._Element]]:
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)
    for index, paragraph in enumerate(paragraphs):
        normalized = re.sub(r"\s+", "", _paragraph_text(paragraph))
        if normalized.startswith("图1功率多普勒成像处理流程图"):
            for candidate in reversed(paragraphs[:index]):
                if candidate.xpath(".//w:drawing", namespaces=NS):
                    captions = [paragraph]
                    if index + 1 < len(paragraphs) and re.match(r"^Fig\.?\s*1\b", _paragraph_text(paragraphs[index + 1]).strip()):
                        captions.append(paragraphs[index + 1])
                    return candidate, captions
        if "图1为CF-sSVD-GPU方法的整体结构" in normalized:
            for candidate in paragraphs[index + 1 :]:
                if candidate.xpath(".//w:drawing", namespaces=NS):
                    return candidate, []
    raise ValueError("未找到图1定位锚点，拒绝回退到全文第一张图片")


def _replace_paragraph_text(paragraph: etree._Element, text: str) -> None:
    text_nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if not text_nodes:
        paragraph.append(_caption_run(text, break_before=False))
        return
    text_nodes[0].text = text
    for node in text_nodes[1:]:
        node.text = ""


def _caption_run(text: str, *, break_before: bool) -> etree._Element:
    run = etree.Element(f"{{{W}}}r")
    props = etree.SubElement(run, f"{{{W}}}rPr")
    fonts = etree.SubElement(props, f"{{{W}}}rFonts")
    fonts.set(f"{{{W}}}ascii", "Times New Roman")
    fonts.set(f"{{{W}}}hAnsi", "Times New Roman")
    fonts.set(f"{{{W}}}eastAsia", "宋体")
    size = etree.SubElement(props, f"{{{W}}}sz")
    size.set(f"{{{W}}}val", "21")
    size_cs = etree.SubElement(props, f"{{{W}}}szCs")
    size_cs.set(f"{{{W}}}val", "21")
    if break_before:
        etree.SubElement(run, f"{{{W}}}br")
    node = etree.SubElement(run, f"{{{W}}}t")
    node.text = text
    return run


def replace_figure1(source_docx: Path, figure_png: Path, output_docx: Path) -> Path:
    source_docx, figure_png, output_docx = map(Path, (source_docx, figure_png, output_docx))
    if source_docx.resolve() == output_docx.resolve():
        raise ValueError("输出路径不得覆盖源DOCX")
    if not source_docx.is_file() or not figure_png.is_file():
        raise FileNotFoundError("源DOCX或图1 PNG不存在")

    with Image.open(figure_png) as image:
        width_px, height_px = image.size
    cx = int(round(FIGURE_WIDTH_MM * EMU_PER_MM))
    cy = int(round(cx * height_px / width_px))

    with ZipFile(source_docx) as source:
        entries = {info.filename: (info, source.read(info.filename)) for info in source.infolist()}

    document_root = etree.fromstring(entries["word/document.xml"][1])
    rels_root = etree.fromstring(entries["word/_rels/document.xml.rels"][1])
    content_types_root = etree.fromstring(entries["[Content_Types].xml"][1])
    figure_paragraph, caption_paragraphs = _find_figure1_paragraph(document_root)
    drawing = figure_paragraph.xpath(".//w:drawing", namespaces=NS)[0]
    blips = drawing.xpath(".//a:blip", namespaces=NS)
    if not blips:
        raise ValueError("图1绘图不含嵌入图片关系")
    figure_rid = blips[0].get(f"{{{R}}}embed")
    relationships = rels_root.xpath(f"./pr:Relationship[@Id='{figure_rid}']", namespaces={"pr": PR})
    if len(relationships) != 1:
        raise ValueError("图1图片关系缺失或不唯一")
    relationship = relationships[0]
    old_target = relationship.get("Target")
    media_name = "word/media/cf_ssvd_gpu_pipeline_figure1.png"
    relationship.set("Target", "media/cf_ssvd_gpu_pipeline_figure1.png")
    for extent in drawing.xpath(".//wp:extent | .//a:xfrm/a:ext", namespaces=NS):
        extent.set("cx", str(cx))
        extent.set("cy", str(cy))

    if not content_types_root.xpath("./ct:Default[@Extension='png']", namespaces={"ct": CT}):
        default = etree.SubElement(content_types_root, f"{{{CT}}}Default")
        default.set("Extension", "png")
        default.set("ContentType", "image/png")

    if caption_paragraphs:
        _replace_paragraph_text(caption_paragraphs[0], ZH_CAPTION)
        if len(caption_paragraphs) > 1:
            _replace_paragraph_text(caption_paragraphs[1], EN_CAPTION)
        else:
            caption_paragraphs[0].addnext(etree.Element(f"{{{W}}}p"))
            new_caption = caption_paragraphs[0].getnext()
            new_caption.append(_caption_run(EN_CAPTION, break_before=False))
    else:
        top_level = drawing
        while top_level.getparent() is not figure_paragraph:
            top_level = top_level.getparent()
        position = figure_paragraph.index(top_level) + 1
        figure_paragraph.insert(position, _caption_run(ZH_CAPTION, break_before=True))
        figure_paragraph.insert(position + 1, _caption_run(EN_CAPTION, break_before=True))

    entries["word/document.xml"] = (
        entries["word/document.xml"][0],
        etree.tostring(document_root, xml_declaration=True, encoding="UTF-8", standalone=True),
    )
    entries["word/_rels/document.xml.rels"] = (
        entries["word/_rels/document.xml.rels"][0],
        etree.tostring(rels_root, xml_declaration=True, encoding="UTF-8", standalone=True),
    )
    entries["[Content_Types].xml"] = (
        entries["[Content_Types].xml"][0],
        etree.tostring(content_types_root, xml_declaration=True, encoding="UTF-8", standalone=True),
    )
    old_media_name = "word/" + old_target if old_target else ""
    old_target_references = sum(1 for rel in rels_root if rel.get("Target") == old_target)
    if old_media_name != media_name and old_target_references == 0:
        entries.pop(old_media_name, None)
    entries[media_name] = (ZipInfo(media_name), figure_png.read_bytes())

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_docx.parent, suffix=".docx", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with ZipFile(temp_path, "w", ZIP_DEFLATED) as target:
            for name, (info, data) in entries.items():
                target.writestr(info, data)
        temp_path.replace(output_docx)
    finally:
        temp_path.unlink(missing_ok=True)
    return output_docx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-docx", type=Path, required=True)
    parser.add_argument("--figure-png", type=Path, required=True)
    parser.add_argument("--output-docx", type=Path, required=True)
    args = parser.parse_args()
    print(replace_figure1(args.source_docx, args.figure_png, args.output_docx))


if __name__ == "__main__":
    main()
