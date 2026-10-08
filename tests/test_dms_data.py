"""Unit tests for DMS annotation normalization and association."""

from __future__ import annotations

import json

import numpy as np

from ultralytics.models.yolo.dms.data import (
    VOCRecord,
    associate_dsm_parts,
    iter_eye12,
    parse_voc_xml,
    rodrigues_to_ypr,
)
from ultralytics.models.yolo.dms.schema import Box, DMSObject


def test_association_uses_geometry_not_xml_order():
    record = VOCRecord(
        annotation=None,  # type: ignore[arg-type] - source path is irrelevant to association
        filename="frame.jpg",
        width=300,
        height=200,
        objects=(
            DMSObject("normalface", Box(150, 0, 300, 200)),
            DMSObject("normalface", Box(0, 0, 140, 200)),
            DMSObject("staring", Box(20, 30, 80, 60)),
            DMSObject("normalmouth", Box(180, 120, 250, 160)),
        ),
    )
    result = associate_dsm_parts(record)
    assert [(item.status, item.face_index, item.target.part) for item in result] == [
        ("matched", 1, "eye"),
        ("matched", 0, "mouth"),
    ]


def test_association_marks_equal_face_candidates_ambiguous():
    record = VOCRecord(
        annotation=None,  # type: ignore[arg-type]
        filename="frame.jpg",
        width=100,
        height=100,
        objects=(
            DMSObject("normalface", Box(0, 0, 100, 100)),
            DMSObject("normalface", Box(0, 0, 100, 100)),
            DMSObject("squint", Box(25, 25, 50, 50)),
        ),
    )
    (result,) = associate_dsm_parts(record)
    assert result.status == "ambiguous"
    assert result.face_index is None


def test_parse_voc_and_eye12(tmp_path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        "<annotation><filename>sample.jpg</filename><size><width>128</width><height>96</height></size>"
        "<object><name>normalface</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>100</xmax><ymax>90</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )
    record = parse_voc_xml(xml)
    assert (record.filename, record.width, record.height, len(record.objects)) == ("sample.jpg", 128, 96, 1)

    jsonl = tmp_path / "eye12.jsonl"
    jsonl.write_text(
        json.dumps({"image": "/tmp/face.jpg", "points": [[x, x + 1] for x in range(12)], "crop_box_xyxy": [0, 0, 128, 128]})
        + "\n",
        encoding="utf-8",
    )
    (eye12,) = tuple(iter_eye12(jsonl))
    assert len(eye12.points) == 12
    assert eye12.crop_box == Box(0.0, 0.0, 128.0, 128.0)


def test_rodrigues_to_ypr_identity_and_yaw():
    assert rodrigues_to_ypr(np.zeros(3)) == (0.0, 0.0, 0.0)
    yaw, pitch, roll = rodrigues_to_ypr(np.array([0.0, np.pi / 2, 0.0]))
    assert np.allclose((yaw, pitch, roll), (90.0, 0.0, 0.0), atol=1e-5)
