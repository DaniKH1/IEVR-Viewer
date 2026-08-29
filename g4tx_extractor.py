"""IEVR Viewer — Level-5 resource viewer and exporter."""

from __future__ import annotations

import io
import math
import os
import copy
import struct
import subprocess
import threading
import sys
import ctypes
import zlib
import json
import webbrowser
import socket
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
import numpy as np
import pyglet
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram


def app_directory() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def bundled_asset(name: str) -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / name


def settings_file() -> Path:
    """Keeps settings beside the executable for a fully portable release."""
    return app_directory() / "IEVR Viewer.settings.json"


def align(value: int, boundary: int) -> int:
    return (value + boundary - 1) & ~(boundary - 1)


def g4sk_payloads(path: Path) -> list[bytes]:
    """Returns standalone or correctly bounded G4SK chunks from a G4PK file."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    if data[:4] == b"G4SK":
        return [data]
    if data[:4] != b"G4PK" or len(data) < 0x30:
        return []
    try:
        header_size = struct.unpack_from("<H", data, 0x04)[0]
        file_count = struct.unpack_from("<I", data, 0x20)[0]
        if not (0 < header_size <= len(data)) or file_count > 10000:
            return []
        offsets = struct.unpack_from(f"<{file_count}I", data, header_size)
        sizes = struct.unpack_from(f"<{file_count}I", data, header_size + file_count * 4)
        payloads = []
        for offset, size in zip(offsets, sizes):
            start = header_size + (offset << 2); end = start + size
            if size >= 0x40 and start >= header_size and end <= len(data) and data[start:start + 4] == b"G4SK":
                payloads.append(data[start:end])
        return payloads
    except (struct.error, MemoryError):
        return []


def g4sk_bones(data: bytes) -> list[dict]:
    """Reads the bone hierarchy used by the original G4 Blender add-on."""
    if data[:4] != b"G4SK" or len(data) < 0x44:
        return []
    try:
        count = struct.unpack_from("<H", data, 0x20)[0]
        if not count or count > 4096 or 0x40 + count * 0x30 > len(data):
            return []
        sections = [0x40 + struct.unpack_from("<H", data, 0x24 + index * 2)[0] * 4 for index in range(8)]
        inverse_bind_table, parent_table, name_table = sections[0], sections[3], sections[7]
        if parent_table + count * 2 > len(data) or name_table + count * 2 > len(data):
            return []
        result = []
        for index in range(count):
            values = struct.unpack_from("<12f", data, 0x40 + index * 0x30)
            # G4SK stores row-major affine matrices, with the translation in
            # the fourth column.  Keep the complete bind pose, not only its
            # position: the FBX skin clusters require both matrices.
            matrix = [*values, 0.0, 0.0, 0.0, 1.0]
            head = (matrix[3], matrix[7], matrix[11])
            inverse_offset = inverse_bind_table + index * 0x30
            if inverse_bind_table > 0 and inverse_offset + 0x30 <= len(data):
                inverse_bind = [*struct.unpack_from("<12f", data, inverse_offset), 0.0, 0.0, 0.0, 1.0]
            else:
                inverse_bind = _rigid_matrix_inverse(matrix)
            parent = struct.unpack_from("<H", data, parent_table + index * 2)[0]
            name_offset = struct.unpack_from("<H", data, name_table + index * 2)[0]
            start = name_table + name_offset; end = data.find(b"\0", start)
            name = data[start:end].decode("ascii", errors="replace") if start < len(data) and end > start else f"Bone_{index:03d}"
            result.append({"name": name or f"Bone_{index:03d}", "head": head, "tail": (head[0], head[1] + .05, head[2]), "parent": parent if parent < count and parent != index else -1, "bind_matrix": matrix, "inverse_bind_matrix": inverse_bind})
        for bone in result:
            parent = bone["parent"]
            if parent >= 0:
                parent_head = result[parent]["head"]
                if sum((bone["head"][i] - parent_head[i]) ** 2 for i in range(3)) > 1e-8:
                    bone["tail"] = parent_head
        return result
    except (struct.error, UnicodeError):
        return []


@dataclass
class Texture:
    name: str
    width: int
    height: int
    payload: bytes


@dataclass
class ModelPreview:
    vertices: list[tuple[float, float, float]]
    uvs: list[tuple[float, float]]
    triangles: list[tuple[int, int, int]]
    triangle_texture_hashes: list[int | None]
    mesh_ranges: list[tuple[str, int, int, int | None]]
    export_meshes: list[dict]
    mesh_count: int


def parse_g4tx(path: Path) -> list[Texture]:
    """Lee la tabla y los payloads de un G4TX, sin alterar el archivo."""
    data = path.read_bytes()
    if len(data) < 0x60 or data[:4] != b"G4TX":
        raise ValueError("Not a valid G4TX file.")
    header_size = struct.unpack_from("<H", data, 0x04)[0]
    table_size = struct.unpack_from("<I", data, 0x0C)[0]
    texture_count = struct.unpack_from("<H", data, 0x20)[0]
    total_count = struct.unpack_from("<H", data, 0x22)[0]
    sub_count = data[0x25]
    if not texture_count or texture_count > total_count:
        raise ValueError("Invalid G4TX texture table.")

    cursor = header_size
    records = []
    for _ in range(texture_count):
        if cursor + 0x30 > len(data):
            raise ValueError("Truncated G4TX file (texture table).")
        values = struct.unpack_from("<IIIIIIHHI", data, cursor)
        records.append((values[1], values[2], values[6], values[7]))
        cursor += 0x30
    cursor += sub_count * 0x18
    cursor = align(cursor, 0x10) + total_count * 4
    cursor = align(cursor + total_count, 4)
    if cursor + total_count * 2 > len(data):
        raise ValueError("Truncated G4TX file (names).")
    offsets = struct.unpack_from(f"<{total_count}H", data, cursor)
    payload_base = align(header_size + table_size, 0x10)

    textures = []
    for index, (offset, size, width, height) in enumerate(records):
        name_start = cursor + offsets[index]
        name_end = data.find(b"\0", name_start)
        if name_end < name_start:
            raise ValueError("Incomplete texture name.")
        name = data[name_start:name_end].decode("ascii", errors="replace") or f"texture_{index:03d}"
        start, end = payload_base + offset, payload_base + offset + size
        if start < payload_base or end > len(data):
            raise ValueError(f"Invalid payload: {name}")
        textures.append(Texture(name, width, height, data[start:end]))
    return textures


# Conversión del almacenamiento Nintendo Switch (NXTCH) a un DDS estándar.
# Es la misma disposición de bloques que usa el extractor del proyecto fuente.
NXTCH_FORMATS = {0x25: (32, 4, None), 0x42: (4, 8, b"DXT1"), 0x44: (8, 16, b"DXT5"), 0x4D: (8, 16, b"DX10")}


def _nx_fields(depth: int, compressed: bool, height: int, extension_count: int) -> list[tuple[int, int]]:
    fields = ({4: [(1, 0), (2, 0), (0, 1), (0, 2), (4, 0), (0, 4), (8, 0), (0, 8), (0, 16), (16, 0)],
               8: [(1, 0), (2, 0), (0, 1), (0, 2), (0, 4), (4, 0), (0, 8), (0, 16), (8, 0)]}
              if compressed else
              {8: [(1, 0), (2, 0), (4, 0), (8, 0), (0, 1), (16, 0), (0, 2), (0, 4), (32, 0)],
               16: [(1, 0), (2, 0), (4, 0), (0, 1), (8, 0), (0, 2), (0, 4), (16, 0)],
               32: [(1, 0), (2, 0), (0, 1), (4, 0), (0, 2), (0, 4), (8, 0)]}).get(depth)
    if fields is None:
        raise ValueError(f"Unsupported NXTCH depth ({depth})")
    fields, start_y, max_size = list(fields), (32 if compressed else 8), (512 if compressed else 128)
    count = extension_count if extension_count >= 0 else 99
    for _ in range(count):
        if start_y >= min(height, max_size):
            break
        fields.append((0, start_y))
        start_y *= 2
    return fields


def _nxtch_level(data: bytes, width: int, height: int, depth: int, unit: int, extension_count: int, compressed: bool) -> bytes:
    fields = _nx_fields(depth, compressed, height, extension_count)
    max_x = max_y = 0
    for x_bit, y_bit in fields:
        max_x |= x_bit
        max_y |= y_bit
    max_x += 1
    max_y += 1
    padded_w = ((width + max_x - 1) // max_x) * max_x
    padded_h = ((height + max_y - 1) // max_y) * max_y
    block_w, block_h = ((width + 3) // 4, (height + 3) // 4) if compressed else (width, height)
    output = bytearray(block_w * block_h * unit)
    per_macro, tile_w = max_x * max_y, (padded_w + max_x - 1) // max_x
    source_count = ((padded_w + 3) // 4) * ((padded_h + 3) // 4) if compressed else padded_w * padded_h
    for source_index in range(min(len(data) // unit, source_count)):
        index = source_index * 16 if compressed else source_index
        macro = index // per_macro
        x, y = (macro % tile_w) * max_x, (macro // tile_w) * max_y
        for bit, (x_bit, y_bit) in enumerate(fields):
            if index >> bit & 1:
                x ^= x_bit
                y ^= y_bit
        if compressed:
            x, y = x // 4, y // 4
        if x < block_w and y < block_h:
            output[(y * block_w + x) * unit:(y * block_w + x + 1) * unit] = data[source_index * unit:(source_index + 1) * unit]
    return bytes(output)


def nxtch_to_dds(payload: bytes) -> bytes:
    if len(payload) < 0x100 or not payload.startswith(b"NXTCH000"):
        raise ValueError("Invalid NXTCH")
    width, height = struct.unpack_from("<II", payload, 0x14)
    fmt, mip_count = struct.unpack_from("<II", payload, 0x24)
    if fmt not in NXTCH_FORMATS or not width or not height or not mip_count:
        raise ValueError(f"Unsupported NXTCH format: 0x{fmt:x}")
    offsets = struct.unpack_from(f"<{mip_count}I", payload, 0x30)
    extension_count = struct.unpack_from("<i", payload, 0x74)[0]
    depth, unit, fourcc = NXTCH_FORMATS[fmt]
    compressed = fmt != 0x25
    levels = []
    for level, offset in enumerate(offsets):
        end = 0x100 + (offsets[level + 1] if level + 1 < mip_count else len(payload) - 0x100)
        levels.append(_nxtch_level(payload[0x100 + offset:end], max(1, width >> level), max(1, height >> level), depth, unit, extension_count, compressed))
    flags, pitch = (0x0002100F if mip_count > 1 else 0x0000100F), (len(levels[0]) if fmt == 0x25 else max(1, (width + 3) // 4) * unit)
    header = bytearray(148 if fmt == 0x4D else 128)
    struct.pack_into("<4s7I", header, 0, b"DDS ", 124, flags, height, width, pitch, 0, mip_count)
    struct.pack_into("<I", header, 76, 32)
    if fourcc:
        struct.pack_into("<I4s", header, 80, 4, fourcc)
    else:
        struct.pack_into("<I4sI4I", header, 80, 0x41, b"\0\0\0\0", 32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000)
    struct.pack_into("<I", header, 108, 0x401008 if mip_count > 1 else 0x1000)
    if fmt == 0x4D:
        struct.pack_into("<5I", header, 128, 98, 3, 0, 1, 0)
    return bytes(header) + b"".join(levels)


def _vertex_layouts(data: bytes, start: int) -> dict[int, dict[int, tuple[int, int]]]:
    """Devuelve {layout: {tipo: (offset, formato)}}. Tipo 10 es UV0."""
    layouts: dict[int, dict[int, tuple[int, int]]] = {}
    cursor = start
    for layout_index in range(data[0x26] if len(data) > 0x26 else 0):
        if cursor + 8 > len(data):
            break
        count = data[cursor + 1]
        end = cursor + 8 + count * 8
        if end > len(data):
            break
        entries: dict[int, tuple[int, int]] = {}
        for index in range(count):
            kind, value_offset, _padding, fmt = struct.unpack_from("<BHBI", data, cursor + 8 + index * 8)
            entries[kind] = (value_offset, fmt)
        layouts[layout_index] = entries
        cursor = end
    return layouts


def _read_uv(raw: bytes, offset: int, format_id: int) -> tuple[float, float]:
    if offset < 0 or offset + 4 > len(raw):
        return 0.0, 0.0
    if format_id in (2, 3) and offset + 8 <= len(raw):
        u, v = struct.unpack_from("<2f", raw, offset)
    elif format_id == 12 and offset + 2 <= len(raw):
        u, v = struct.unpack_from("<2B", raw, offset)
        u, v = u / 255.0, v / 255.0
    elif format_id == 14 and offset + 4 <= len(raw):
        u, v = struct.unpack_from("<2H", raw, offset)
        u, v = u / 65535.0, v / 65535.0
    elif format_id in (18, 20) and offset + 4 <= len(raw):
        u, v = struct.unpack_from("<2h", raw, offset)
        u, v = u / 32767.0, v / 32767.0
    else:
        u, v = struct.unpack_from("<2H", raw, offset)
        u, v = u / 65535.0, v / 65535.0
    # Sólo V se adapta a la carga de imagen. U permanece en el espacio UV
    # nativo; el reflejo necesario pertenece a la cámara del visor.
    return float(u), 1.0 - float(v)


def _g4md_material_texture_hashes(data: bytes, mesh_table: int, mesh_count: int, material_count: int) -> dict[int, int]:
    """Resuelve material -> CRC de textura desde las tablas G4MD."""
    cursor = mesh_table + mesh_count * 0x50
    for _ in range(data[0x26] if len(data) > 0x26 else 0):
        if cursor + 8 > len(data):
            return {}
        cursor += 8 + data[cursor + 1] * 8
    material_table = align(cursor, 0x10)
    if material_count <= 0 or material_table + material_count * 0x10 > len(data):
        return {}
    # La tabla de slots está después de los bloques de parámetros. Localizamos
    # su firma: entradas de seis bytes cuyo segundo byte es el slot difuso 3.
    fallback = material_table + material_count * 0x10 + 0x30
    best, best_run = fallback, -1
    for candidate in range(align(fallback, 0x10), min(len(data) - 6, fallback + 0x1000) + 1, 0x10):
        run = 0
        while candidate + run * 6 + 6 <= len(data):
            off = candidate + run * 6
            if data[off + 1] != 3 or data[off] > 0x40:
                break
            run += 1
        if run > best_run:
            best, best_run = candidate, run
    base_bias = struct.unpack_from("<H", data, 0x0A)[0] * 4
    hash_table = base_bias + struct.unpack_from("<H", data, 0x76)[0] * 4
    hash_count = data[0x27] if len(data) > 0x27 else 0
    if hash_table <= 0 or hash_table + hash_count * 4 > len(data):
        return {}
    hashes = struct.unpack_from(f"<{hash_count}I", data, hash_table)
    result: dict[int, int] = {}
    for material in range(material_count):
        off = material_table + material * 0x10
        count = data[off + 0x0C]
        start = struct.unpack_from("<H", data, off + 0x0E)[0]
        for item in range(count):
            ref = best + (start + item) * 6
            if ref + 6 <= len(data) and data[ref + 1] == 3 and data[ref] < len(hashes):
                result[material] = hashes[data[ref]]
                break
    return result


def _g4md_name(data: bytes, table: int, index: int, fallback: str) -> str:
    if table <= 0 or table + index * 2 + 2 > len(data):
        return fallback
    offset = struct.unpack_from("<H", data, table + index * 2)[0]
    start = table + offset
    end = data.find(b"\0", start)
    if start >= len(data) or end <= start:
        return fallback
    return data[start:end].decode("ascii", errors="replace") or fallback


def _read_vertex_influences(data: bytes, start: int, stride: int, layout: dict[int, tuple[int, int]], palette: list[int]) -> list[tuple[int, float]]:
    weight_offset = layout.get(5, (0x24, 0))[0]
    joint_offset = layout.get(6, (0x34, 0))[0]
    if start + max(weight_offset + 16, joint_offset + 8) > len(data) or start + stride > len(data):
        return []
    # A layout element ends where the next element begins.  Reading to the end
    # of a vertex record blends unrelated attributes into the armature weights.
    offsets = sorted(offset for offset, _format in layout.values() if 0 <= offset < stride)
    weight_end = next((offset for offset in offsets if offset > weight_offset), stride)
    joint_end = next((offset for offset in offsets if offset > joint_offset), stride)
    max_count = min(8, (weight_end - weight_offset) // 2, joint_end - joint_offset)
    if max_count <= 0:
        return []
    weights = struct.unpack_from(f"<{max_count}H", data, start + weight_offset)
    joints = struct.unpack_from(f"<{max_count}B", data, start + joint_offset)
    total = sum(weight for joint, weight in zip(joints, weights) if joint != 0xFF)
    if not total:
        return []
    return [((palette[joint] if 0 <= joint < len(palette) else joint), weight / total)
            for joint, weight in zip(joints, weights) if joint != 0xFF and weight]


def _fbx_name(value: str) -> str:
    """Makes an FBX-safe name while retaining the original mesh/bone name."""
    return value.replace("\\", "_").replace('"', "'").replace("\x00", "") or "Unnamed"


def _fbx_numbers(values, per_line: int = 16) -> str:
    """Formats an FBX ASCII array without locale-dependent decimal commas."""
    rendered = []
    for value in values:
        if isinstance(value, float):
            rendered.append(format(value, ".9g"))
        else:
            rendered.append(str(int(value)))
    if not rendered:
        return ""
    return ",".join(rendered)


def _fbx_array(name: str, values, indent: str = "        ") -> list[str]:
    values = list(values)
    return [f"{indent}{name}: *{len(values)} {{", f"{indent}    a: {_fbx_numbers(values)}", f"{indent}}}"]


def _matrix_at(position: tuple[float, float, float]) -> list[float]:
    # FBX arrays are column-major.  This is an identity matrix translated to
    # ``position`` after being serialized in the layout FBX expects.
    x, y, z = position
    return [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., x, y, z, 1.]


def _rigid_matrix_inverse(matrix: list[float]) -> list[float]:
    """Inverse of the orthonormal G4SK affine matrix (row-major)."""
    r00, r01, r02, tx = matrix[0], matrix[1], matrix[2], matrix[3]
    r10, r11, r12, ty = matrix[4], matrix[5], matrix[6], matrix[7]
    r20, r21, r22, tz = matrix[8], matrix[9], matrix[10], matrix[11]
    return [
        r00, r10, r20, -(r00 * tx + r10 * ty + r20 * tz),
        r01, r11, r21, -(r01 * tx + r11 * ty + r21 * tz),
        r02, r12, r22, -(r02 * tx + r12 * ty + r22 * tz),
        0.0, 0.0, 0.0, 1.0,
    ]


def _fbx_matrix_from_g4(matrix: list[float] | None) -> list[float]:
    """Serializes a row-major G4SK matrix in the FBX column-major layout."""
    if matrix is None or len(matrix) != 16:
        return _matrix_at((0.0, 0.0, 0.0))
    return [matrix[row * 4 + column] for column in range(4) for row in range(4)]


def _g4_to_blender_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Converts G4's Y-up coordinates to Blender's Z-up coordinates."""
    x, y, z = point
    return x, -z, y


def _g4_to_blender_matrix(matrix: list[float] | None) -> list[float] | None:
    """Changes the basis of a row-major G4SK affine matrix (Y-up -> Z-up)."""
    if matrix is None or len(matrix) != 16:
        return matrix
    rotation = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    inverse = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    def multiply(left: list[float], right: list[float]) -> list[float]:
        return [sum(left[row * 4 + index] * right[index * 4 + column] for index in range(4)) for row in range(4) for column in range(4)]
    return multiply(multiply(rotation, matrix), inverse)


def _smooth_vertex_normals(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> list[float]:
    """Creates one averaged normal per vertex, without FBX sharp-edge data."""
    accum = [[0.0, 0.0, 0.0] for _ in vertices]
    for face in faces:
        a, b, c = (vertices[index] for index in face)
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        normal = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        for index in face:
            accum[index][0] += normal[0]; accum[index][1] += normal[1]; accum[index][2] += normal[2]
    values = []
    for x, y, z in accum:
        length = math.sqrt(x * x + y * y + z * z) or 1.0
        values.extend((x / length, y / length, z / length))
    return values


def write_native_fbx(destination: Path, meshes: list[dict], bones: list[dict], textures: dict[int, Path]) -> None:
    """Writes a self-contained ASCII FBX with meshes, skin clusters and UVs.

    This intentionally uses only the Python standard library.  Every source
    mesh stays separate; skin clusters represent the source vertex groups.
    """
    next_id = 100000

    def allocate() -> int:
        nonlocal next_id
        next_id += 1
        return next_id

    root_id = allocate()
    bone_ids = [allocate() for _ in bones]
    mesh_records = []
    for index, mesh in enumerate(meshes):
        mesh_records.append({"mesh": mesh, "model": allocate(), "geometry": allocate(), "skin": allocate(), "index": index})
    material_ids: dict[int, int] = {key: allocate() for key in textures}
    texture_ids: dict[int, int] = {key: allocate() for key in textures}
    video_ids: dict[int, int] = {key: allocate() for key in textures}

    lines = [
        "; FBX 7.4.0 project generated by IEVR Viewer", "; Native export: Blender is not required.",
        "FBXHeaderExtension:  {", "    FBXHeaderVersion: 1003", "    FBXVersion: 7400",
        "    Creator: \"IEVR Viewer\"", "}",
        "GlobalSettings:  {", "    Version: 1000", "    Properties70:  {",
        "        P: \"UpAxis\", \"int\", \"Integer\", \"\",1", "        P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1",
        "        P: \"FrontAxis\", \"int\", \"Integer\", \"\",2", "        P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",-1",
        "        P: \"CoordAxis\", \"int\", \"Integer\", \"\",0", "        P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1",
        "        P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",100", "        P: \"OriginalUnitScaleFactor\", \"double\", \"Number\", \"\",100", "    }", "}",
        "Definitions:  {", "    Version: 100", "    Count: 7", "    ObjectType: \"Model\" { Count: 0 }", "}", "Objects:  {",
        f"    Model: {root_id}, \"Model::G4_Armature\", \"Null\" {{", "        Version: 232", "    }",
    ]
    for index, bone in enumerate(bones):
        name = _fbx_name(bone["name"])
        head = _g4_to_blender_point(bone["head"])
        parent = bone.get("parent", -1)
        if 0 <= parent < len(bones):
            base = _g4_to_blender_point(bones[parent]["head"])
            translation = (head[0] - base[0], head[1] - base[1], head[2] - base[2])
        else:
            translation = head
        length = max(.001, math.dist(head, _g4_to_blender_point(bone.get("tail", (bone["head"][0], bone["head"][1] + .05, bone["head"][2])))))
        lines.extend([f"    Model: {bone_ids[index]}, \"Model::{name}\", \"LimbNode\" {{", "        Version: 232", "        Properties70:  {",
                      f"            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",{translation[0]:.9g},{translation[1]:.9g},{translation[2]:.9g}",
                      f"            P: \"LimbLength\", \"double\", \"Number\", \"\",{length:.9g}", "        }", "    }"])
    for record in mesh_records:
        mesh = record["mesh"]
        vertices = [coordinate for vertex in mesh["vertices"] for coordinate in _g4_to_blender_point(vertex)]
        polygon_indices = []
        uv = []
        uv_indices = []
        for face in mesh["faces"]:
            for vertex_index in face:
                polygon_indices.append(vertex_index)
                u, v = mesh["uvs"][vertex_index]
                uv.extend((u, v)); uv_indices.append(len(uv_indices))
            polygon_indices[-1] = -polygon_indices[-1] - 1
        name = _fbx_name(mesh["name"])
        lines.extend([f"    Geometry: {record['geometry']}, \"Geometry::{name}\", \"Mesh\" {{", "        GeometryVersion: 124"])
        lines.extend(_fbx_array("Vertices", vertices))
        lines.extend(_fbx_array("PolygonVertexIndex", polygon_indices))
        lines.extend(["        LayerElementNormal: 0 {", "            Version: 101", "            Name: \"\"", "            MappingInformationType: \"ByVertice\"", "            ReferenceInformationType: \"Direct\""])
        # One normal per vertex gives Blender smooth shading and deliberately
        # emits no edge-smoothing/sharp-edge layer.
        normals = _smooth_vertex_normals([_g4_to_blender_point(vertex) for vertex in mesh["vertices"]], mesh["faces"])
        lines.extend(_fbx_array("Normals", normals, "            "))
        lines.extend(["        }", "        LayerElementUV: 0 {", "            Version: 101", "            Name: \"UVMap\"", "            MappingInformationType: \"ByPolygonVertex\"", "            ReferenceInformationType: \"IndexToDirect\""])
        lines.extend(_fbx_array("UV", uv, "            ")); lines.extend(_fbx_array("UVIndex", uv_indices, "            "))
        lines.extend(["        }", "        LayerElementMaterial: 0 {", "            Version: 101", "            MappingInformationType: \"AllSame\"", "            ReferenceInformationType: \"IndexToDirect\"", "            Materials: *1 { a: 0 }", "        }", "        LayerElementSmoothing: 0 {", "            Version: 102", "            Name: \"\"", "            MappingInformationType: \"ByPolygon\"", "            ReferenceInformationType: \"Direct\""])
        lines.extend(_fbx_array("Smoothing", [1] * len(mesh["faces"]), "            "))
        lines.extend(["        }", "        Layer: 0 {", "            Version: 100", "            LayerElement:  { Type: \"LayerElementNormal\" TypedIndex: 0 }", "            LayerElement:  { Type: \"LayerElementUV\" TypedIndex: 0 }", "            LayerElement:  { Type: \"LayerElementMaterial\" TypedIndex: 0 }", "            LayerElement:  { Type: \"LayerElementSmoothing\" TypedIndex: 0 }", "        }", "    }"])
        lines.extend([f"    Model: {record['model']}, \"Model::{name}\", \"Mesh\" {{", "        Version: 232", "        Properties70:  { P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1 }", "    }"])
        lines.extend([f"    Deformer: {record['skin']}, \"Deformer::Skin_{name}\", \"Skin\" {{", "        Version: 101", "        Link_DeformAcuracy: 50", "    }"])
        # Create a cluster for each source bone that has actual non-zero weights.
        for bone_index, bone in enumerate(bones):
            indices, weights = [], []
            for vertex_index, influences in enumerate(mesh.get("influences", [])):
                for influenced_bone, weight in influences:
                    if influenced_bone == bone_index and weight > 0:
                        indices.append(vertex_index); weights.append(weight)
            if indices:
                cluster_id = allocate(); record.setdefault("clusters", []).append((cluster_id, bone_index))
                lines.extend([f"    Deformer: {cluster_id}, \"SubDeformer::{_fbx_name(bone['name'])}\", \"Cluster\" {{", "        Version: 100", "        UserData: \"\",", "        LinkMode: \"Normalize\""])
                lines.extend(_fbx_array("Indexes", indices)); lines.extend(_fbx_array("Weights", weights))
                bind_matrix = bone.get("bind_matrix")
                inverse_bind = bone.get("inverse_bind_matrix") or (_rigid_matrix_inverse(bind_matrix) if bind_matrix else None)
                # Transform is the inverse bind matrix in bone space; Link is
                # the full world bind pose.  Using only bone heads here made
                # every skinned mesh receive an incorrect extra transform.
                lines.extend(_fbx_array("Transform", _fbx_matrix_from_g4(_g4_to_blender_matrix(inverse_bind))))
                lines.extend(_fbx_array("TransformLink", _fbx_matrix_from_g4(_g4_to_blender_matrix(bind_matrix))))
                lines.extend(_fbx_array("TransformAssociateModel", _matrix_at((0., 0., 0.))))
                lines.extend(["    }"])
    for key, texture_path in textures.items():
        safe = _fbx_name(texture_path.stem); file_name = str(texture_path).replace("\\", "/")
        lines.extend([f"    Material: {material_ids[key]}, \"Material::Mat_{safe}\", \"\" {{", "        Version: 102", "        ShadingModel: \"phong\"", "        MultiLayer: 0", "        Properties70:  { P: \"DiffuseColor\", \"Color\", \"\", \"A\",1,1,1; P: \"DiffuseFactor\", \"Number\", \"\", \"A\",1; P: \"TransparencyFactor\", \"Number\", \"\", \"A\",0 }", "    }"])
        lines.extend([f"    Video: {video_ids[key]}, \"Video::{safe}\", \"Clip\" {{", "        Type: \"Clip\"", f"        FileName: \"{file_name}\"", f"        RelativeFilename: \"{texture_path.name}\"", "    }"])
        lines.extend([f"    Texture: {texture_ids[key]}, \"Texture::{safe}\", \"TextureVideoClip\" {{", "        Type: \"TextureVideoClip\"", "        Version: 202", f"        FileName: \"{file_name}\"", f"        RelativeFilename: \"{texture_path.name}\"", "        ModelUVTranslation: 0,0", "        ModelUVScaling: 1,1", "        Texture_Alpha_Source: \"None\"", "        Cropping: 0,0,0,0", "    }"])
    lines.extend(["}", "Connections:  {"])
    for index, bone in enumerate(bones):
        parent = bones[index].get("parent", -1)
        lines.append(f"    C: \"OO\",{bone_ids[index]},{bone_ids[parent] if 0 <= parent < len(bones) else root_id}")
    for record in mesh_records:
        mesh = record["mesh"]; texture_hash = mesh.get("texture_hash")
        lines.extend([f"    C: \"OO\",{record['geometry']},{record['model']}", f"    C: \"OO\",{record['model']},0", f"    C: \"OO\",{record['skin']},{record['geometry']}"])
        if texture_hash in material_ids:
            lines.extend([f"    C: \"OO\",{material_ids[texture_hash]},{record['model']}", f"    C: \"OP\",{texture_ids[texture_hash]},{material_ids[texture_hash]},\"DiffuseColor\"", f"    C: \"OO\",{video_ids[texture_hash]},{texture_ids[texture_hash]}"])
        for cluster_id, bone_index in record.get("clusters", []):
            lines.extend([f"    C: \"OO\",{cluster_id},{record['skin']}", f"    C: \"OO\",{bone_ids[bone_index]},{cluster_id}"])
    lines.extend(["}", "Takes:  { Current: \"\" }"])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _FbxNode:
    def __init__(self, name: str, properties=(), children=()):
        self.name = name.encode("utf-8")
        self.properties = tuple(properties)
        self.children = list(children)


def _fbx_binary_property(value) -> bytes:
    """Encodes one typed FBX 7.4 property (the format Unity/Blender import)."""
    if isinstance(value, tuple) and len(value) == 2 and value[0] in {"d", "f", "i", "l"}:
        kind, items = value[0], list(value[1]); format_char = {"d": "d", "f": "f", "i": "i", "l": "q"}[kind]
        raw = struct.pack("<" + format_char * len(items), *items) if items else b""
        return kind.encode("ascii") + struct.pack("<III", len(items), 0, len(raw)) + raw
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + struct.pack("<I", len(raw)) + raw
    if isinstance(value, bool):
        return b"C" + bytes((1 if value else 0,))
    if isinstance(value, int):
        return b"L" + struct.pack("<q", value)
    return b"D" + struct.pack("<d", float(value))


def _fbx_binary_node(node: _FbxNode, start: int) -> bytes:
    props = b"".join(_fbx_binary_property(value) for value in node.properties)
    header_length = 13 + len(node.name)
    child_start = start + header_length + len(props)
    children = b""; position = child_start
    for child in node.children:
        encoded = _fbx_binary_node(child, position)
        children += encoded; position += len(encoded)
    body = node.name + props + children + (b"\0" * 13 if node.children else b"")
    return struct.pack("<IIIB", start + 13 + len(body), len(node.properties), len(props), len(node.name)) + body


def _fbx_p(name: str, value, value_type: str = "double") -> _FbxNode:
    property_class = {"double": "Number", "int": "Integer"}.get(value_type, "")
    property_flag = "" if value_type in {"double", "int"} else "A"
    if isinstance(value, tuple):
        return _FbxNode("P", (name, value_type, property_class, property_flag, *value))
    return _FbxNode("P", (name, value_type, property_class, property_flag, value))


def _fbx_object_name(name: str, object_class: str) -> str:
    # Binary FBX stores the display name and its class in one string, separated
    # by the official NUL/SOH marker (not by the ASCII-only "Model::" prefix).
    return f"{name}\0\x01{object_class}"


def _fbx_model_properties(translation=(0., 0., 0.)) -> _FbxNode:
    return _FbxNode("Properties70", children=[
        _fbx_p("Lcl Translation", translation, "Lcl Translation"),
        _fbx_p("Lcl Rotation", (0., 0., 0.), "Lcl Rotation"),
        _fbx_p("Lcl Scaling", (1., 1., 1.), "Lcl Scaling"),
        _fbx_p("Visibility", 1., "Visibility"), _fbx_p("DefaultAttributeIndex", 0, "int"), _fbx_p("InheritType", 1, "int"),
    ])


def write_native_binary_fbx(destination: Path, meshes: list[dict], bones: list[dict], textures: dict[int, Path]) -> None:
    """Writes binary FBX 7.4 without Blender or any external executable."""
    next_id = 100000
    def allocate() -> int:
        nonlocal next_id
        next_id += 1
        return next_id
    root_id = allocate(); root_attribute_id = allocate(); bone_ids = [allocate() for _ in bones]; bone_attribute_ids = [allocate() for _ in bones]
    records = [{"mesh": mesh, "model": allocate(), "geometry": allocate(), "skin": allocate()} for mesh in meshes]
    material_ids = {key: allocate() for key in textures}; texture_ids = {key: allocate() for key in textures}; video_ids = {key: allocate() for key in textures}
    objects = [_FbxNode("NodeAttribute", (root_attribute_id, _fbx_object_name("G4_Armature", "NodeAttribute"), "Null"), []), _FbxNode("Model", (root_id, _fbx_object_name("G4_Armature", "Model"), "Null"), [_FbxNode("Version", (232,)), _fbx_model_properties()])]
    for index, bone in enumerate(bones):
        parent = bone.get("parent", -1); head = _g4_to_blender_point(bone["head"])
        parent_head = _g4_to_blender_point(bones[parent]["head"]) if 0 <= parent < len(bones) else None
        translation = tuple(head[i] - parent_head[i] for i in range(3)) if parent_head is not None else head
        length = max(.001, math.dist(head, _g4_to_blender_point(bone.get("tail", (bone["head"][0], bone["head"][1] + .05, bone["head"][2])))))
        bone_name = _fbx_name(bone["name"])
        objects.append(_FbxNode("NodeAttribute", (bone_attribute_ids[index], _fbx_object_name(bone_name, "NodeAttribute"), "LimbNode"), [_FbxNode("TypeFlags", ("Skeleton",)), _FbxNode("Properties70", children=[_fbx_p("Size", max(.001, length))])]))
        model_props = _fbx_model_properties(translation); model_props.children.append(_fbx_p("LimbLength", length))
        objects.append(_FbxNode("Model", (bone_ids[index], _fbx_object_name(bone_name, "Model"), "LimbNode"), [_FbxNode("Version", (232,)), model_props]))
    for record in records:
        mesh = record["mesh"]; name = _fbx_name(mesh["name"])
        transformed_vertices = [_g4_to_blender_point(vertex) for vertex in mesh["vertices"]]
        vertices = [number for vertex in transformed_vertices for number in vertex]; indices=[]; uv=[]; uv_indices=[]
        for face in mesh["faces"]:
            for vertex_index in face:
                indices.append(vertex_index); point_uv=mesh["uvs"][vertex_index]; uv.extend(point_uv); uv_indices.append(len(uv_indices))
            indices[-1] = -indices[-1]-1
        normals = _smooth_vertex_normals(transformed_vertices, mesh["faces"])
        geometry_children = [_FbxNode("GeometryVersion", (124,)), _FbxNode("Vertices", (("d", vertices),)), _FbxNode("PolygonVertexIndex", (("i", indices),)),
            _FbxNode("LayerElementNormal", (0,), [_FbxNode("Version", (101,)), _FbxNode("Name", ("",)), _FbxNode("MappingInformationType", ("ByVertice",)), _FbxNode("ReferenceInformationType", ("Direct",)), _FbxNode("Normals", (("d", normals),))]),
            _FbxNode("LayerElementUV", (0,), [_FbxNode("Version", (101,)), _FbxNode("Name", ("UVMap",)), _FbxNode("MappingInformationType", ("ByPolygonVertex",)), _FbxNode("ReferenceInformationType", ("IndexToDirect",)), _FbxNode("UV", (("d", uv),)), _FbxNode("UVIndex", (("i", uv_indices),))]),
            _FbxNode("LayerElementMaterial", (0,), [_FbxNode("Version", (101,)), _FbxNode("Name", ("",)), _FbxNode("MappingInformationType", ("AllSame",)), _FbxNode("ReferenceInformationType", ("IndexToDirect",)), _FbxNode("Materials", (("i", [0]),))]),
            _FbxNode("LayerElementSmoothing", (0,), [_FbxNode("Version", (102,)), _FbxNode("Name", ("",)), _FbxNode("MappingInformationType", ("ByPolygon",)), _FbxNode("ReferenceInformationType", ("Direct",)), _FbxNode("Smoothing", (("i", [1] * len(mesh["faces"])),))]),
            _FbxNode("Layer", (0,), [_FbxNode("Version", (100,)), _FbxNode("LayerElement", children=[_FbxNode("Type", ("LayerElementNormal",)), _FbxNode("TypedIndex", (0,))]), _FbxNode("LayerElement", children=[_FbxNode("Type", ("LayerElementUV",)), _FbxNode("TypedIndex", (0,))]), _FbxNode("LayerElement", children=[_FbxNode("Type", ("LayerElementMaterial",)), _FbxNode("TypedIndex", (0,))]), _FbxNode("LayerElement", children=[_FbxNode("Type", ("LayerElementSmoothing",)), _FbxNode("TypedIndex", (0,))])])]
        objects.append(_FbxNode("Geometry", (record["geometry"], _fbx_object_name(name, "Geometry"), "Mesh"), geometry_children))
        objects.append(_FbxNode("Model", (record["model"], _fbx_object_name(name, "Model"), "Mesh"), [_FbxNode("Version", (232,)), _fbx_model_properties()]))
        objects.append(_FbxNode("Deformer", (record["skin"], _fbx_object_name(f"Skin_{name}", "Deformer"), "Skin"), [_FbxNode("Version", (101,)), _FbxNode("Link_DeformAcuracy", (50.,))]))
        for bone_index, bone in enumerate(bones):
            weights = [(vertex_index, weight) for vertex_index, influence in enumerate(mesh.get("influences", [])) for influenced_bone, weight in influence if influenced_bone == bone_index and weight > 0]
            if not weights: continue
            cluster = allocate(); record.setdefault("clusters", []).append((cluster, bone_index))
            bind_matrix = bone.get("bind_matrix")
            inverse_bind = bone.get("inverse_bind_matrix") or (_rigid_matrix_inverse(bind_matrix) if bind_matrix else None)
            cluster_children = [
                _FbxNode("Version", (100,)), _FbxNode("UserData", ("",)),
                _FbxNode("Indexes", (("i", [item[0] for item in weights]),)),
                _FbxNode("Weights", (("d", [item[1] for item in weights]),)),
                _FbxNode("Transform", (("d", _fbx_matrix_from_g4(_g4_to_blender_matrix(inverse_bind))),)),
                _FbxNode("TransformLink", (("d", _fbx_matrix_from_g4(_g4_to_blender_matrix(bind_matrix))),)),
                _FbxNode("TransformAssociateModel", (("d", _matrix_at((0., 0., 0.))),)),
                _FbxNode("LinkMode", ("Normalize",)),
            ]
            objects.append(_FbxNode("Deformer", (cluster, _fbx_object_name(_fbx_name(bone['name']), "SubDeformer"), "Cluster"), cluster_children))
    for key, texture_path in textures.items():
        name = _fbx_name(texture_path.stem); absolute = str(texture_path.resolve()).replace("\\", "/")
        objects.extend([_FbxNode("Material", (material_ids[key], _fbx_object_name(f"Mat_{name}", "Material"), ""), [_FbxNode("Version", (102,)), _FbxNode("ShadingModel", ("phong",)), _FbxNode("MultiLayer", (0,)), _FbxNode("Properties70", children=[_fbx_p("DiffuseColor", (1.,1.,1.), "Color"), _fbx_p("DiffuseFactor", 1.), _fbx_p("TransparencyFactor", 0.)])]), _FbxNode("Video", (video_ids[key], _fbx_object_name(name, "Video"), "Clip"), [_FbxNode("Type", ("Clip",)), _FbxNode("FileName", (absolute,)), _FbxNode("RelativeFilename", (texture_path.name,))]), _FbxNode("Texture", (texture_ids[key], _fbx_object_name(name, "Texture"), "TextureVideoClip"), [_FbxNode("Type", ("TextureVideoClip",)), _FbxNode("Version", (202,)), _FbxNode("FileName", (absolute,)), _FbxNode("RelativeFilename", (texture_path.name,)), _FbxNode("ModelUVTranslation", (0.,0.)), _FbxNode("ModelUVScaling", (1.,1.)), _FbxNode("Texture_Alpha_Source", ("None",)), _FbxNode("Cropping", (0,0,0,0))])])
    connections = [_FbxNode("C", ("OO", root_id, 0)), _FbxNode("C", ("OO", root_attribute_id, root_id))]
    for index, bone in enumerate(bones):
        parent = bone.get("parent", -1)
        connections.extend([_FbxNode("C", ("OO", bone_attribute_ids[index], bone_ids[index])), _FbxNode("C", ("OO", bone_ids[index], bone_ids[parent] if 0 <= parent < len(bones) else root_id))])
    for record in records:
        texture_hash = record["mesh"].get("texture_hash"); connections.extend([_FbxNode("C", ("OO", record["geometry"], record["model"])), _FbxNode("C", ("OO", record["model"], 0)), _FbxNode("C", ("OO", record["skin"], record["geometry"]))])
        if texture_hash in material_ids: connections.extend([_FbxNode("C", ("OO", material_ids[texture_hash], record["model"])), _FbxNode("C", ("OP", texture_ids[texture_hash], material_ids[texture_hash], "DiffuseColor")), _FbxNode("C", ("OO", video_ids[texture_hash], texture_ids[texture_hash]))])
        for cluster, bone_index in record.get("clusters", []): connections.extend([_FbxNode("C", ("OO", cluster, record["skin"])), _FbxNode("C", ("OO", bone_ids[bone_index], cluster))])
    # G4 coordinates are already in Blender-sized units.  FBX uses centimetres
    # for its unit metadata, so declaring 100 avoids Blender importing every
    # object with a 0.01 object scale.
    top = [_FbxNode("FBXHeaderExtension", children=[_FbxNode("FBXHeaderVersion", (1003,)), _FbxNode("FBXVersion", (7400,)), _FbxNode("Creator", ("IEVR Viewer",))]), _FbxNode("GlobalSettings", children=[_FbxNode("Version", (1000,)), _FbxNode("Properties70", children=[_fbx_p("UpAxis", 1, "int"), _fbx_p("UpAxisSign", 1, "int"), _fbx_p("FrontAxis", 2, "int"), _fbx_p("FrontAxisSign", -1, "int"), _fbx_p("CoordAxis", 0, "int"), _fbx_p("CoordAxisSign", 1, "int"), _fbx_p("UnitScaleFactor", 100.), _fbx_p("OriginalUnitScaleFactor", 100.)])]), _FbxNode("Definitions", children=[_FbxNode("Version", (100,)), _FbxNode("Count", (0,))]), _FbxNode("Objects", children=objects), _FbxNode("Connections", children=connections), _FbxNode("Takes", children=[_FbxNode("Current", ("",))])]
    header = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400)
    payload = b""; position = len(header)
    for node in top:
        encoded = _fbx_binary_node(node, position); payload += encoded; position += len(encoded)
    destination.write_bytes(header + payload + b"\0" * 13)


def read_g4md_preview(path: Path) -> ModelPreview:
    """Lee posiciones e índices de G4MD/G4MG para una vista previa ligera."""
    if path.suffix.lower() != ".g4md":
        raise ValueError("The built-in 3D viewer supports .g4md models with their matching .g4mg file.")
    data = path.read_bytes()
    companion = path.with_suffix(".g4mg")
    if data[:4] != b"G4MD":
        raise ValueError("The selected file does not have a valid G4MD header.")
    if not companion.is_file():
        raise ValueError(f"Missing matching geometry file: {companion.name}")
    geometry = companion.read_bytes()
    if len(data) < 0x60:
        raise ValueError("Incomplete G4MD file.")
    table_offset = struct.unpack_from("<H", data, 0x04)[0]
    mesh_count = struct.unpack_from("<H", data, 0x20)[0]
    material_count = struct.unpack_from("<H", data, 0x22)[0]
    index_base = struct.unpack_from("<I", data, 0x5C)[0]
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_texture_hashes: list[int | None] = []
    mesh_ranges: list[tuple[str, int, int, int | None]] = []
    export_meshes: list[dict] = []
    layouts = _vertex_layouts(data, table_offset + mesh_count * 0x50)
    material_hashes = _g4md_material_texture_hashes(data, table_offset, mesh_count, material_count)
    name_bias = struct.unpack_from("<H", data, 0x0A)[0] * 4
    mesh_name_table = name_bias + struct.unpack_from("<H", data, 0x84)[0] * 4
    palette_table = name_bias + struct.unpack_from("<H", data, 0x82)[0] * 4
    for mesh_index in range(mesh_count):
        offset = table_offset + mesh_index * 0x50
        if offset + 0x50 > len(data):
            continue
        vertex_offset, index_offset, vertex_count, index_count = struct.unpack_from("<4I", data, offset)
        triangle_count = struct.unpack_from("<I", data, offset + 0x30)[0]
        stride = struct.unpack_from("<H", data, offset + 0x3E)[0] & 0xFF or 0x44
        flags0 = struct.unpack_from("<H", data, offset + 0x3C)[0]
        palette_start = struct.unpack_from("<H", data, offset + 0x3A)[0]
        layout_index, material_index = struct.unpack_from("<BB", data, offset + 0x42)
        layout = layouts.get(layout_index, {})
        uv_offset, uv_format = layout.get(10, (0x40 if stride >= 0x44 else 0x20, 14))
        count = index_count or triangle_count * 3
        if not vertex_count or vertex_offset + vertex_count * stride > len(geometry):
            continue
        if index_base + index_offset + count * 2 > len(geometry):
            continue
        base = len(vertices)
        triangle_start = len(triangles)
        local_vertices, local_uvs, local_influences, local_triangles = [], [], [], []
        palette_length = flags0 & 0xFF if flags0 & 0x100 else 0
        palette = list(struct.unpack_from(f"<{palette_length}H", data, palette_table + palette_start * 2)) if palette_length and palette_table + (palette_start + palette_length) * 2 <= len(data) else []
        for vertex_index in range(vertex_count):
            vertex_start = vertex_offset + vertex_index * stride
            position = struct.unpack_from("<3f", geometry, vertex_start)
            uv = _read_uv(geometry, vertex_start + uv_offset, uv_format)
            vertices.append(position); uvs.append(uv)
            local_vertices.append(position); local_uvs.append(uv)
            local_influences.append(_read_vertex_influences(geometry, vertex_start, stride, layout, palette))
        indices = struct.unpack_from(f"<{count}H", geometry, index_base + index_offset)
        for index in range(0, len(indices) - 2, 3):
            a, b, c = indices[index:index + 3]
            if a < vertex_count and b < vertex_count and c < vertex_count:
                # G4 conserva el orden opuesto al frontal que espera el visor
                # y el exportador FBX nativo.
                triangles.append((base + a, base + b, base + c))
                triangle_texture_hashes.append(material_hashes.get(material_index))
                local_triangles.append((a, b, c))
        if len(triangles) > triangle_start:
            mesh_name = _g4md_name(data, mesh_name_table, mesh_index, f"Mesh_{mesh_index:03d}")
            mesh_ranges.append((mesh_name, triangle_start, len(triangles), material_hashes.get(material_index)))
            export_meshes.append({"name": mesh_name, "vertices": local_vertices, "uvs": local_uvs, "faces": local_triangles,
                                  "influences": local_influences, "joint_palette": palette, "texture_hash": material_hashes.get(material_index)})
    if not vertices or not triangles:
        raise ValueError("No previewable geometry was found in the G4MD/G4MG pair.")
    return ModelPreview(vertices, uvs, triangles, triangle_texture_hashes, mesh_ranges, export_meshes, mesh_count)


def run_gpu_viewer(model_path: Path, variants: list[Path], initial_variant: int, initial_mode: str, parent_hwnd: int | None = None, control_port: int | None = None) -> None:
    """Visor G4MD acelerado por GPU. Nunca reduce ni salta triángulos."""
    model = read_g4md_preview(model_path)
    positions = np.asarray(model.vertices, dtype=np.float32)
    center = positions.mean(axis=0)
    radius = float(np.linalg.norm(positions - center, axis=1).max()) or 1.0
    positions = (positions - center) / radius
    vertex_positions = positions
    vertex_uvs = np.asarray(model.uvs, dtype=np.float32)
    vertex_shader = """#version 330 core
in vec3 position; in vec2 tex_coords; out vec2 uv;
uniform float yaw; uniform float pitch; uniform float zoom; uniform float aspect;
void main() { vec3 p=position; float a=cos(yaw),b=sin(yaw),c=cos(pitch),d=sin(pitch);
 p.xz=vec2(p.x*a-p.z*b,p.x*b+p.z*a); p.yz=vec2(p.y*c-p.z*d,p.y*d+p.z*c);
 // Los ejes de cámara G4 y OpenGL difieren horizontalmente; reflejar la
 // proyección, no las UV, preserva la orientación de la textura.
 p.x*=-aspect; gl_Position=vec4(p*zoom,1.0); uv=tex_coords; }"""
    fragment_shader = """#version 330 core
in vec2 uv; out vec4 color; uniform sampler2D tex; uniform int mode;
void main() { if(mode==2) color=texture(tex,uv); else { float light=.72; color=vec4(.30*light,.62*light,.85*light,1.0); } }"""
    window = pyglet.window.Window(1000, 760, caption=f"Visor G4MD — {model_path.name}", resizable=True, vsync=True)
    if parent_hwnd:
        # Pyglet crea una ventana OpenGL nativa; en Windows puede vivir como
        # hija del Canvas de Tk, conservando el render GPU dentro de la pestaña.
        user32 = ctypes.windll.user32
        hwnd = window._hwnd
        user32.SetParent(hwnd, parent_hwnd)
        style = user32.GetWindowLongW(hwnd, -16)
        user32.SetWindowLongW(hwnd, -16, (style & ~0x00C00000) | 0x40000000 | 0x10000000)
        window.set_caption("")

        def fit_to_parent(_dt: float) -> None:
            rect = wintypes.RECT()
            if not user32.GetClientRect(parent_hwnd, ctypes.byref(rect)):
                return
            width, height = max(1, rect.right - rect.left), max(1, rect.bottom - rect.top)
            if window.width != width or window.height != height:
                window.set_size(width, height)
                # Reafirma posición y orden tras cada cambio de panel.
                user32.SetWindowPos(hwnd, 0, 0, 0, width, height, 0x0001 | 0x0004 | 0x0010)
        fit_to_parent(0)
        pyglet.clock.schedule_interval(fit_to_parent, .03)
    program = ShaderProgram(Shader(vertex_shader, "vertex"), Shader(fragment_shader, "fragment"))
    def build_vertex_lists(source: ModelPreview) -> dict:
        positions = np.asarray(source.vertices, dtype=np.float32)
        center = positions.mean(axis=0)
        radius = float(np.linalg.norm(positions - center, axis=1).max()) or 1.0
        positions = (positions - center) / radius
        uvs = np.asarray(source.uvs, dtype=np.float32)
        grouped_indices: dict[int | None, list[int]] = {}
        for triangle, texture_hash in zip(source.triangles, source.triangle_texture_hashes):
            grouped_indices.setdefault(texture_hash, []).extend(triangle)
        result = {}
        for texture_hash, indices in grouped_indices.items():
            local = np.asarray(indices, dtype=np.uint32)
            result[texture_hash] = program.vertex_list(len(local), gl.GL_TRIANGLES,
                position=("f", positions[local].reshape(-1).tolist()), tex_coords=("f", uvs[local].reshape(-1).tolist()))
        return result

    vertex_lists = build_vertex_lists(model)
    # Los G4MD se almacenan orientados al reverso respecto a la cámara.
    state = {"yaw": math.pi - .55, "pitch": .25, "zoom": .72, "mode": initial_mode, "variant": initial_variant, "drag": None, "textures": {}, "fallback": None}

    def load_texture() -> None:
        state["textures"] = {}
        state["fallback"] = None
        if not variants:
            window.set_caption(f"G4MD Viewer — {model_path.name} (no G4TX)")
            return
        try:
            # El G4MD puede apuntar a mapas repartidos entre varias variantes.
            # La seleccionada tiene prioridad; las demás completan referencias
            # que no estén presentes en ella.
            ordered = [variants[state["variant"]]] + [path for index, path in enumerate(variants) if index != state["variant"]]
            for container in ordered:
                for texture in parse_g4tx(container):
                    name = texture.name.lower()
                    if name.endswith(("line", "msk", "oc", "sp", "spm", "nml", ".2", "_n", "_nm")):
                        continue
                    texture_hash = zlib.crc32(texture.name.encode("ascii", errors="replace")) & 0xFFFFFFFF
                    if texture_hash in state["textures"]:
                        continue
                    payload = nxtch_to_dds(texture.payload) if texture.payload.startswith(b"NXTCH000") else texture.payload
                    image = Image.open(io.BytesIO(payload)).convert("RGBA")
                    gl_texture = pyglet.image.ImageData(image.width, image.height, "RGBA", image.tobytes(), pitch=-image.width * 4).get_texture()
                    state["textures"][texture_hash] = gl_texture
                    if state["fallback"] is None:
                        state["fallback"] = gl_texture
            window.set_caption(f"G4MD Viewer — {model_path.name} — {state['variant'] + 1}/{len(variants)} {variants[state['variant']].name}")
        except Exception:
            window.set_caption(f"G4MD Viewer — {model_path.name} (unsupported texture)")

    control_socket = None
    if control_port is not None:
        try:
            control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            control_socket.bind(("127.0.0.1", control_port))
            control_socket.listen()
            control_socket.setblocking(False)
        except OSError:
            control_socket = None

    def poll_controls(_dt: float) -> None:
        nonlocal model, model_path, variants, vertex_lists
        if control_socket is None:
            return
        while True:
            try:
                client, _address = control_socket.accept()
            except BlockingIOError:
                break
            try:
                payload = {}
                payload = json.loads(client.recv(1_000_000).decode("utf-8"))
                command = payload.get("command")
                if command == "mode":
                    state["mode"] = payload.get("mode", state["mode"])
                elif command == "variant" and variants:
                    state["variant"] = max(0, min(int(payload.get("variant", 0)), len(variants) - 1)); load_texture()
                elif command == "model":
                    replacement = read_g4md_preview(Path(payload["path"]))
                    for vertex_list in vertex_lists.values():
                        vertex_list.delete()
                    model = replacement; model_path = Path(payload["path"]); variants = [Path(item) for item in payload.get("variants", [])]
                    state["variant"] = max(0, min(int(payload.get("variant", 0)), max(0, len(variants) - 1)))
                    state["mode"] = payload.get("mode", state["mode"]); vertex_lists = build_vertex_lists(model); load_texture()
                if payload.get("wait"):
                    client.sendall(b"OK")
            except Exception:
                if payload.get("wait"):
                    try: client.sendall(b"ERROR")
                    except OSError: pass
            finally:
                client.close()

    @window.event
    def on_draw() -> None:
        gl.glClearColor(30 / 255, 37 / 255, 48 / 255, 1.0)
        window.clear()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if state["mode"] == "wireframe" else gl.GL_FILL)
        program.use()
        program["yaw"] = state["yaw"]; program["pitch"] = state["pitch"]; program["zoom"] = state["zoom"]
        program["aspect"] = window.height / max(1, window.width)
        program["mode"] = 2 if state["mode"] == "textured" and state["fallback"] else 1
        for texture_hash, vertex_list in vertex_lists.items():
            texture = state["textures"].get(texture_hash, state["fallback"])
            if texture:
                gl.glActiveTexture(gl.GL_TEXTURE0); texture.bind(); program["tex"] = 0
            vertex_list.draw(gl.GL_TRIANGLES)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers) -> None:
        if buttons & pyglet.window.mouse.LEFT:
            state["yaw"] -= dx * .012; state["pitch"] = max(-1.5, min(1.5, state["pitch"] + dy * .012))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y) -> None:
        state["zoom"] = max(.08, min(4., state["zoom"] * (1.12 if scroll_y > 0 else .89)))

    @window.event
    def on_key_press(symbol, modifiers) -> None:
        if symbol == pyglet.window.key.W: state["mode"] = "wireframe"
        elif symbol == pyglet.window.key.S: state["mode"] = "solid"
        elif symbol == pyglet.window.key.T: state["mode"] = "textured"
        elif variants and symbol in (pyglet.window.key.LEFT, pyglet.window.key.RIGHT):
            state["variant"] = (state["variant"] + (-1 if symbol == pyglet.window.key.LEFT else 1)) % len(variants); load_texture()

    load_texture()
    pyglet.clock.schedule_interval(poll_controls, .05)
    pyglet.app.run()


class G4TXExtractor(tk.Tk):
    THUMB_SIZE = 154
    MODEL_EXTENSIONS = {".g4md", ".g4pkm", ".objbin"}

    def __init__(self) -> None:
        super().__init__()
        self.title("IEVR Viewer")
        try:
            self.app_icon = tk.PhotoImage(file=str(bundled_asset("app_icon.png")))
            self.iconphoto(True, self.app_icon)
        except tk.TclError:
            self.app_icon = None
        self.geometry("1200x760")
        self.minsize(900, 580)
        self.root_path: Path | None = None
        self.current_file: Path | None = None
        self.textures: list[Texture] = []
        self.selected: set[int] = set()
        self.thumbnails: list[ImageTk.PhotoImage] = []
        self.show_models = tk.BooleanVar(value=False)
        self.model_root_path: Path | None = None
        self.preview_model: ModelPreview | None = None
        self.model_yaw = -0.55
        self.model_pitch = 0.25
        self.model_zoom = 1.0
        self.model_drag_at: tuple[int, int] | None = None
        self.model_view_mode = "textured"
        self.model_path: Path | None = None
        self.model_variants: list[Path] = []
        self.model_variant_index = 0
        self.model_texture_image: ImageTk.PhotoImage | None = None
        self.model_texture: Image.Image | None = None
        self.model_render_image: ImageTk.PhotoImage | None = None
        self.model_render_after: str | None = None
        self.viewer_process: subprocess.Popen | None = None
        self.viewer_control_port: int | None = None
        self.model_load_token = 0
        self.model_loading = False
        self.skeletons_by_stem: dict[str, list[Path]] = {}
        self.skeleton_files: list[Path] = []
        self.export_in_progress = False
        self.export_lod_mode = tk.StringVar(value="Include LOD meshes")
        self.export_progress_text = tk.StringVar(value="")
        self.status = tk.StringVar(value="Choose a folder to search for .g4tx files.")
        self._build_ui()
        # Give Tk time to map both tabs before populating their trees.  This
        # keeps saved folders visible immediately on slower machines too.
        self.after(200, self.restore_settings)

    def save_settings(self) -> None:
        try:
            settings_file().write_text(json.dumps({"g4tx_folder": str(self.root_path or ""), "g4md_folder": str(self.model_root_path or "")}, indent=2), encoding="utf-8")
        except OSError:
            pass

    def restore_settings(self) -> None:
        path = settings_file()
        try:
            if not path.is_file():
                self.status.set("No saved folder settings beside IEVR Viewer.exe yet.")
                return
            values = json.loads(path.read_text(encoding="utf-8"))
            g4tx = Path(values.get("g4tx_folder", ""))
            g4md = Path(values.get("g4md_folder", ""))
            if g4tx.is_dir():
                self.root_path = g4tx
                self.populate_tree()
            if g4md.is_dir():
                self.model_root_path = g4md
                self.populate_model_tree()
            if g4tx.is_dir() or g4md.is_dir():
                self.status.set("Restored folders from the previous session.")
            else:
                self.status.set("Saved folders are no longer available. Choose their new location to update the settings.")
        except (OSError, ValueError, TypeError) as exc:
            self.status.set(f"Could not restore the saved folders: {exc}")

    def show_credits(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("IEVR Viewer — Credits")
        dialog.transient(self)
        dialog.resizable(False, False)
        content = ttk.Frame(dialog, padding=20)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="IEVR Viewer", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(content, text="Idea Designer: DaniKH").pack(anchor="w", pady=(14, 0))
        ttk.Label(content, text="Original Blender Add-on Author: The-RealBobi").pack(anchor="w", pady=(8, 0))
        link = ttk.Label(content, text="github.com/The-RealBobi/G4_Blender", foreground="#1769aa", cursor="hand2")
        link.pack(anchor="w", pady=(2, 0))
        link.bind("<Button-1>", lambda _event: webbrowser.open_new("https://github.com/The-RealBobi/G4_Blender"))
        ttk.Label(content, text="Built with Codex").pack(anchor="w", pady=(16, 0))
        ttk.Button(content, text="Close", command=dialog.destroy).pack(anchor="e", pady=(18, 0))

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="G4TX Folder…", command=self.choose_folder).pack(side="left")
        ttk.Button(toolbar, text="G4MD Folder…", command=self.choose_model_folder).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Credits", command=self.show_credits).pack(side="right")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        texture_tab = ttk.Frame(notebook)
        model_tab = ttk.Frame(notebook)
        help_tab = ttk.Frame(notebook)
        notebook.add(texture_tab, text="G4TX Textures")
        notebook.add(model_tab, text="G4MD Models")
        notebook.add(help_tab, text="Instructions")
        self._build_instructions_tab(help_tab)
        texture_actions = ttk.Frame(texture_tab, padding=(4, 4, 4, 0))
        texture_actions.pack(fill="x")
        ttk.Button(texture_actions, text="Export Selected to PNG…", command=self.export_selected).pack(side="left")
        ttk.Button(texture_actions, text="Select All", command=lambda: self.set_all(True)).pack(side="left", padx=(16, 0))
        ttk.Button(texture_actions, text="Clear Selection", command=lambda: self.set_all(False)).pack(side="left", padx=(8, 0))
        body = ttk.Panedwindow(texture_tab, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=4)
        body.add(left, weight=1)
        body.add(right, weight=4)

        ttk.Label(left, text="G4TX Files", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse")
        tree_bar = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_bar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.open_from_tree)

        self.heading = ttk.Label(right, text="No file selected", font=("Segoe UI", 11, "bold"))
        self.heading.pack(anchor="w", pady=(0, 6))
        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.grid = ttk.Frame(self.canvas, padding=6)
        self.grid.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.grid, anchor="nw")
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        # La rueda se reserva para el visor 3D: las texturas G4TX no deben
        # saltar de posición al pasar el ratón sobre la cuadrícula.
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=(8, 4)).pack(fill="x", side="bottom")
        self._build_model_tab(model_tab)

    def _build_instructions_tab(self, parent: ttk.Frame) -> None:
        """Short in-app guide, kept available without opening a browser."""
        guide = tk.Text(parent, wrap="word", borderwidth=0, padx=22, pady=18, font=("Segoe UI", 10))
        guide.pack(fill="both", expand=True)
        guide.insert("1.0", """IEVR Viewer — Quick Guide

1. G4TX Textures
Choose G4TX Folder… to scan a directory and all its subfolders. Expand folders in the tree, select a .g4tx file, choose the maps you want, then use Export Selected to PNG….

2. G4MD Models
Choose G4MD Folder… and select a .g4md model. The viewer supports Wireframe, Solid and Textured modes. Drag with the left mouse button to rotate and use the mouse wheel to zoom. If the G4TX and G4MD folders share the same structure, matching textures are detected automatically.

3. Texture variants
Use the ◀ and ▶ buttons above the viewer to switch between G4TX variants located in the matching folder. Exporting to FBX extracts every texture from the variant currently selected in the viewer.

4. Exporting FBX
Select Main meshes only to omit LOD meshes, or Include LOD meshes to export them too. The FBX includes separate meshes, UVs, materials, the armature and vertex groups. Its textures are exported into a companion _textures folder and that folder opens when exporting finishes.

5. Saved folders
IEVR Viewer.settings.json is stored beside IEVR Viewer.exe. Keep it alongside the executable to reopen your selected G4TX and G4MD folders automatically next time.
""")
        guide.configure(state="disabled")

    def _build_model_tab(self, parent: ttk.Frame) -> None:
        actions = ttk.Frame(parent, padding=(4, 4, 4, 0))
        actions.pack(fill="x")
        self.export_fbx_button = ttk.Button(actions, text="Export to FBX…", command=self.export_model_fbx)
        self.export_fbx_button.pack(side="left")
        ttk.Label(actions, text="Export:").pack(side="left", padx=(12, 4))
        self.export_lod_selector = ttk.Combobox(
            actions, textvariable=self.export_lod_mode, state="readonly", width=19,
            values=("Main meshes only", "Include LOD meshes"),
        )
        self.export_lod_selector.pack(side="left")
        self.export_progress = ttk.Progressbar(actions, mode="determinate", length=160, maximum=100)
        self.export_progress.pack(side="left", padx=(10, 0))
        self.export_progress_label = ttk.Label(actions, textvariable=self.export_progress_text)
        self.export_progress_label.pack(side="left", padx=(6, 0))
        self.model_load_progress = ttk.Progressbar(actions, mode="indeterminate", length=145)
        self.model_load_label = ttk.Label(actions, text="")
        self.model_load_label.pack(side="left", padx=(12, 0))
        self.model_load_progress.pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Wireframe", command=lambda: self.set_model_view("wireframe")).pack(side="left", padx=(16, 0))
        ttk.Button(actions, text="Solid", command=lambda: self.set_model_view("solid")).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Textured", command=lambda: self.set_model_view("textured")).pack(side="left", padx=(8, 0))
        variant_actions = ttk.Frame(actions)
        variant_actions.pack(side="right")
        ttk.Button(variant_actions, text="◀", command=lambda: self.change_model_variant(-1)).pack(side="left")
        self.model_variant_label = ttk.Label(variant_actions, text="No G4TX variant")
        self.model_variant_label.pack(side="left", padx=8)
        ttk.Button(variant_actions, text="▶", command=lambda: self.change_model_variant(1)).pack(side="left")
        body = ttk.Panedwindow(parent, orient="horizontal")
        body.pack(fill="both", expand=True, padx=4, pady=4)
        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=4)
        body.add(left, weight=1)
        body.add(right, weight=4)
        ttk.Label(left, text="G4MD Files", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.model_tree = ttk.Treeview(left, show="tree", selectmode="browse")
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=scrollbar.set)
        self.model_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.model_tree.bind("<<TreeviewSelect>>", self.open_model_from_tree)
        self.model_tree.bind("<Button-3>", self.show_model_context_menu)
        self.model_context_menu = tk.Menu(self, tearoff=False)
        self.model_context_menu.add_command(label="Open file path", command=self.open_selected_model_path)

        self.model_heading = ttk.Label(right, text="Choose a model folder", font=("Segoe UI", 11, "bold"))
        self.model_heading.pack(anchor="w", pady=(0, 6))
        self.model_info = ttk.Label(right, text="", foreground="#666")
        self.model_info.pack(anchor="w", pady=(0, 6))
        self.model_canvas = tk.Canvas(right, background="#1e2530", highlightthickness=0)
        self.model_canvas.pack(fill="both", expand=True)
        self.model_canvas.bind("<Configure>", lambda _event: self.schedule_model_render())
        self.model_canvas.bind("<ButtonPress-1>", self.begin_model_drag)
        self.model_canvas.bind("<B1-Motion>", self.drag_model)
        self.model_canvas.bind("<MouseWheel>", self.zoom_model)

    def choose_model_folder(self) -> None:
        picked = filedialog.askdirectory(title="Choose the model root folder")
        if not picked:
            return
        self.model_root_path = Path(picked)
        self.populate_model_tree()
        self.save_settings()

    def populate_model_tree(self) -> None:
        self.model_tree.delete(*self.model_tree.get_children())
        if self.model_root_path is None:
            return
        self.skeletons_by_stem = {}
        self.skeleton_files = []
        folders: dict[Path, str] = {self.model_root_path: ""}
        models = 0
        unreadable: list[str] = []

        def report_error(error: OSError) -> None:
            if error.filename:
                unreadable.append(error.filename)

        for current_name, directory_names, file_names in os.walk(self.model_root_path, onerror=report_error):
            current = Path(current_name)
            directory_names.sort(key=str.casefold)
            file_names.sort(key=str.casefold)
            if current not in folders:
                parent_id = folders.get(current.parent, "")
                folders[current] = self.model_tree.insert(parent_id, "end", text=current.name, open=False)
            current_id = folders[current]
            for directory_name in directory_names:
                child = current / directory_name
                if child not in folders:
                    folders[child] = self.model_tree.insert(current_id, "end", text=directory_name, open=False)
            for file_name in file_names:
                path = current / file_name
                if path.suffix.lower() == ".g4sk":
                    self.skeletons_by_stem.setdefault(path.stem.casefold(), []).append(path)
                    self.skeleton_files.append(path)
                if path.suffix.lower() in self.MODEL_EXTENSIONS:
                    self.model_tree.insert(current_id, "end", text=file_name, values=(str(path),), tags=("model",))
                    models += 1
        self.status.set(f"Found {models} 3D model(s) in {len(folders) - 1} subfolder(s).")
        if unreadable:
            self.status.set(self.status.get() + f" Could not read {len(unreadable)} folder(s).")
        self.model_heading.configure(text="No compatible models found" if not models else "Select a model to preview it")

    def open_model_from_tree(self, _event: tk.Event) -> None:
        if self.model_loading:
            return
        selected = self.model_tree.selection()
        if not selected:
            return
        values = self.model_tree.item(selected[0], "values")
        if values:
            self.load_model_preview(Path(values[0]))

    def show_model_context_menu(self, event: tk.Event) -> None:
        item = self.model_tree.identify_row(event.y)
        if not item or not self.model_tree.item(item, "values"):
            return
        self.model_tree.selection_set(item)
        self.model_context_menu.tk_popup(event.x_root, event.y_root)

    def open_selected_model_path(self) -> None:
        selected = self.model_tree.selection()
        if not selected:
            return
        values = self.model_tree.item(selected[0], "values")
        if values:
            subprocess.Popen(["explorer", "/select," + str(Path(values[0]))])

    def load_model_preview(self, path: Path) -> None:
        if self.model_loading:
            return
        self.model_load_token += 1
        token = self.model_load_token
        self.model_loading = True
        self.model_tree.state(["disabled"])
        self.model_load_label.configure(text="Loading model…")
        self.model_load_progress.configure(mode="indeterminate")
        self.model_load_progress.start(12)
        self.model_heading.configure(text=f"Loading {path.name}…")
        self.model_info.configure(text="Reading geometry in the background…")
        self.status.set(f"Loading {path.name}…")
        threading.Thread(target=self._read_model_preview_worker, args=(path, token), daemon=True).start()

    def _read_model_preview_worker(self, path: Path, token: int) -> None:
        try:
            preview = read_g4md_preview(path)
            self.after(0, lambda: self._apply_model_preview(path, preview, token))
        except Exception as exc:
            self.after(0, lambda: self._show_model_preview_error(path, exc, token))

    def _apply_model_preview(self, path: Path, preview: ModelPreview, token: int) -> None:
        if token != self.model_load_token:
            return
        try:
            self.preview_model = preview
            self.model_path = path
            self.refresh_model_variants()
            self.model_yaw, self.model_pitch, self.model_zoom = math.pi - .55, 0.25, 1.0
            self.model_heading.configure(text=path.name)
            self.model_info.configure(
                text=f"{self.preview_model.mesh_count} mesh(es) · {len(self.preview_model.vertices):,} vertices · "
                     f"{len(self.preview_model.triangles):,} triangles  |  GPU Viewer: W wireframe · S solid · T textured"
            )
            self.model_canvas.delete("all")
            self.model_load_label.configure(text="Updating GPU viewer…")
            threading.Thread(target=self._replace_gpu_model_worker, args=(token,), daemon=True).start()
        except Exception as exc:
            self._show_model_preview_error(path, exc, token)

    def _show_model_preview_error(self, path: Path, exc: Exception, token: int) -> None:
        if token != self.model_load_token:
            return
        self.preview_model = None
        self.model_heading.configure(text=f"Could not preview {path.name}")
        self.model_info.configure(text=str(exc))
        self.render_model_preview()
        self._finish_model_load(token)

    def _finish_model_load(self, token: int) -> None:
        if token != self.model_load_token:
            return
        self.model_loading = False
        self.model_load_progress.stop()
        self.model_load_progress.configure(value=0)
        self.model_load_label.configure(text="")
        self.model_tree.state(["!disabled"])

    def _replace_gpu_model_worker(self, token: int) -> None:
        if token != self.model_load_token or self.model_path is None:
            return
        payload = {"command": "model", "path": str(self.model_path), "variants": [str(path) for path in self.model_variants], "variant": self.model_variant_index, "mode": self.model_view_mode}
        if self.send_viewer_command(payload, wait_for_reply=True):
            self.after(0, lambda: self._complete_gpu_model_replace(token))
        else:
            self.after(0, lambda: self._launch_gpu_model_replace(token))

    def _complete_gpu_model_replace(self, token: int) -> None:
        if token != self.model_load_token:
            return
        self.status.set("GPU preview loaded. No model faces are omitted.")
        self._finish_model_load(token)

    def _launch_gpu_model_replace(self, token: int) -> None:
        if token != self.model_load_token:
            return
        self.open_gpu_viewer()
        self.after(800, lambda: self._complete_gpu_model_replace(token))

    def begin_model_drag(self, event: tk.Event) -> None:
        self.model_drag_at = (event.x, event.y)

    def drag_model(self, event: tk.Event) -> None:
        if self.model_drag_at is None:
            return
        old_x, old_y = self.model_drag_at
        self.model_yaw += (event.x - old_x) * 0.012
        self.model_pitch = max(-1.45, min(1.45, self.model_pitch + (event.y - old_y) * 0.012))
        self.model_drag_at = (event.x, event.y)
        self.schedule_model_render()

    def zoom_model(self, event: tk.Event) -> None:
        self.model_zoom = max(0.12, min(8.0, self.model_zoom * (1.12 if event.delta > 0 else 0.89)))
        self.schedule_model_render()

    def set_model_view(self, mode: str) -> None:
        self.model_view_mode = mode
        if not self.send_viewer_command({"command": "mode", "mode": mode}):
            self.open_gpu_viewer()

    def refresh_model_variants(self) -> None:
        self.model_variants = []
        self.model_variant_index = 0
        if self.model_path is not None and self.root_path is not None and self.model_root_path is not None:
            try:
                relative = self.model_path.parent.relative_to(self.model_root_path)
                candidate = self.root_path / relative
                self.model_variants = sorted((item for item in candidate.iterdir() if item.suffix.lower() == ".g4tx"), key=lambda item: item.name.casefold())
            except (ValueError, OSError):
                pass
        self.update_model_variant_preview()

    def change_model_variant(self, direction: int) -> None:
        if not self.model_variants:
            return
        self.model_variant_index = (self.model_variant_index + direction) % len(self.model_variants)
        self.update_model_variant_preview()
        if not self.send_viewer_command({"command": "variant", "variant": self.model_variant_index}):
            self.open_gpu_viewer()

    def send_viewer_command(self, payload: dict, wait_for_reply: bool = False) -> bool:
        if self.viewer_process is None or self.viewer_process.poll() is not None or self.viewer_control_port is None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", self.viewer_control_port), timeout=30 if wait_for_reply else .2) as client:
                request = dict(payload)
                request["wait"] = wait_for_reply
                client.sendall(json.dumps(request).encode("utf-8"))
                if wait_for_reply:
                    client.settimeout(30)
                    return client.recv(16) == b"OK"
            return True
        except OSError:
            return False

    def open_gpu_viewer(self) -> None:
        if self.model_path is None:
            return
        if self.model_render_after is not None:
            self.after_cancel(self.model_render_after)
            self.model_render_after = None
        payload = {"command": "model", "path": str(self.model_path), "variants": [str(path) for path in self.model_variants], "variant": self.model_variant_index, "mode": self.model_view_mode}
        if self.send_viewer_command(payload):
            return
        if self.viewer_process is not None and self.viewer_process.poll() is None:
            self.viewer_process.terminate()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            self.viewer_control_port = probe.getsockname()[1]
        parent_hwnd = self.model_canvas.winfo_id()
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--viewer", str(self.model_path)]
        else:
            command = [sys.executable, str(Path(__file__).resolve()), "--viewer", str(self.model_path)]
        command.extend(str(path) for path in self.model_variants)
        command.extend(["--variant", str(self.model_variant_index), "--mode", self.model_view_mode, "--parent", str(parent_hwnd), "--control-port", str(self.viewer_control_port)])
        self.viewer_process = subprocess.Popen(command, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def update_model_variant_preview(self) -> None:
        self.model_texture_image = None
        self.model_texture = None
        if not self.model_variants:
            self.model_variant_label.configure(text="No G4TX variant")
            self.render_model_preview()
            return
        variant = self.model_variants[self.model_variant_index]
        self.model_variant_label.configure(text=f"{self.model_variant_index + 1}/{len(self.model_variants)}  {variant.name}")
        # The GPU child process decodes the full texture set itself. Avoiding a
        # duplicate texture decode in Tk keeps model changes responsive.

    def export_model_fbx(self) -> None:
        if self.preview_model is None or self.model_path is None:
            messagebox.showinfo("Export FBX", "Select a previewable G4MD first.")
            return
        if self.export_in_progress:
            return
        destination = filedialog.asksaveasfilename(title="Export to FBX", defaultextension=".fbx", filetypes=[("FBX", "*.fbx")])
        if not destination:
            return
        meshes = copy.deepcopy(self.preview_model.export_meshes)
        include_lods = self.export_lod_mode.get() == "Include LOD meshes"
        if not include_lods:
            meshes = [mesh for mesh in meshes if "_lod" not in mesh.get("name", "").casefold()]
        if not meshes:
            messagebox.showwarning("Export FBX", "This model has no main meshes to export.")
            return
        self.export_in_progress = True
        self.export_fbx_button.configure(state="disabled")
        self.export_progress.configure(value=0)
        self.export_progress_text.set("Preparing export…")
        threading.Thread(target=self._export_model_fbx_worker, args=(Path(destination), meshes, self.model_path, list(self.model_variants), self.model_variant_index, include_lods), daemon=True).start()

    def _set_export_progress(self, value: int, text: str) -> None:
        self.export_progress.configure(value=value)
        self.export_progress_text.set(text)

    def _finish_export(self, message: str | None = None, error: Exception | None = None) -> None:
        self.export_in_progress = False
        self.export_fbx_button.configure(state="normal")
        if error is not None:
            self.export_progress_text.set("Export failed")
            messagebox.showerror("Export FBX", str(error))
            return
        self.export_progress.configure(value=100)
        self.export_progress_text.set("Export complete")
        if message:
            self.status.set(message)

    def _export_model_fbx_worker(self, destination: Path, meshes: list[dict], model_path: Path, variants: list[Path], selected_variant: int, include_lods: bool) -> None:
        output_folder = destination.parent
        texture_folder = output_folder / f"{Path(destination).stem}_textures"
        exported_textures: dict[int, Path] = {}
        # Blender's FBX importer automatically wires alpha whenever the diffuse
        # image is RGBA.  Diffuse maps used by this FBX are deliberately saved
        # as RGB, while auxiliary maps keep their original alpha channels.
        opaque_material_hashes = {mesh.get("texture_hash") for mesh in meshes if mesh.get("texture_hash") is not None}
        try:
            texture_folder.mkdir(parents=True, exist_ok=True)
            self.after(0, lambda: self._set_export_progress(8, "Exporting…"))
            if variants:
                # Export every map in the G4TX variant selected in the viewer,
                # including masks, normals and auxiliary maps.  They are useful
                # outside Blender even when the FBX only assigns its diffuse map.
                selected_container = variants[max(0, min(selected_variant, len(variants) - 1))]
                for texture in parse_g4tx(selected_container):
                        texture_hash = zlib.crc32(texture.name.encode("ascii", errors="replace")) & 0xFFFFFFFF
                        if texture_hash in exported_textures:
                            continue
                        payload = nxtch_to_dds(texture.payload) if texture.payload.startswith(b"NXTCH000") else texture.payload
                        target = texture_folder / f"{texture.name}.png"
                        image = Image.open(io.BytesIO(payload))
                        image.convert("RGB" if texture_hash in opaque_material_hashes else "RGBA").save(target, "PNG")
                        exported_textures[texture_hash] = target
                self.after(0, lambda: self._set_export_progress(58, "Exporting…"))
            self.after(0, lambda: self._set_export_progress(62, "Finding skeleton and vertex groups…"))
            bones = self.read_export_skeleton(model_path)
            self.remap_export_skin_to_skeleton(bones, meshes, model_path)
            self.after(0, lambda: self._set_export_progress(78, "Writing FBX…"))
            write_native_binary_fbx(destination, meshes, bones, exported_textures)
            weight_count = sum(len(influences) for mesh in meshes for influences in mesh.get("influences", []))
            lod_text = "including LOD meshes" if include_lods else "main meshes only"
            message = f"Exported native FBX ({lod_text}): {len(meshes)} mesh(es), {len(bones)} bone(s), {weight_count} weight(s), and {len(exported_textures)} texture(s)."
            self.after(0, lambda: self._finish_export(message))
            subprocess.Popen(["explorer", str(output_folder)])
        except Exception as exc:
            self.after(0, lambda: self._finish_export(error=Exception(f"Could not export the FBX:\n{exc}")))

    def remap_export_skin_to_skeleton(self, bones: list[dict], meshes: list[dict], model_path: Path) -> None:
        """Maps the G4MD joint palette to the actual G4SK bone order by CRC32 name."""
        if not bones:
            return
        try:
            data = model_path.read_bytes()
            name_bias = struct.unpack_from("<H", data, 0x0A)[0] * 4
            hash_table = name_bias + struct.unpack_from("<H", data, 0x74)[0] * 4
            if hash_table <= 0 or hash_table >= len(data):
                return
            by_hash = {zlib.crc32(bone["name"].encode("ascii", errors="replace")) & 0xFFFFFFFF: index for index, bone in enumerate(bones)}
            for mesh in meshes:
                if mesh.get("_skin_remapped"):
                    continue
                remapped = []
                for influences in mesh.get("influences", []):
                    converted = []
                    for source_joint, weight in influences:
                        value_offset = hash_table + source_joint * 4
                        matching_bone = None
                        if 0 <= source_joint and value_offset + 4 <= len(data):
                            matching_bone = by_hash.get(struct.unpack_from("<I", data, value_offset)[0])
                        # Some simple assets use direct G4SK indexes rather than
                        # a CRC palette.  Keep that valid representation too.
                        if matching_bone is None and 0 <= source_joint < len(bones):
                            matching_bone = source_joint
                        if matching_bone is not None:
                            converted.append((matching_bone, weight))
                    remapped.append(converted)
                mesh["influences"] = remapped
                mesh["_skin_remapped"] = True
        except (OSError, struct.error):
            return

    def read_export_skeleton(self, model_path: Path | None = None) -> list[dict]:
        """Finds the G4SK whose bone names best match the model's joint hashes."""
        model_path = model_path or self.model_path
        if model_path is None:
            return []
        candidates = [model_path.with_suffix(".g4sk"), model_path.parent / f"{model_path.parent.name}.g4sk"]
        candidates.extend(self.skeletons_by_stem.get(model_path.stem.casefold(), []))
        candidates.extend(sorted(model_path.parent.glob("*.g4sk")))
        candidates.extend(sorted(model_path.parent.glob("*.g4pk")))
        candidates.extend(sorted(model_path.parent.glob("*.g4pkm")))
        candidates.extend(self.skeleton_files)
        # De-duplicate while preserving nearby/exact files as tie breakers.
        unique_candidates = list(dict.fromkeys(path for path in candidates if path.is_file()))
        try:
            data = model_path.read_bytes()
            name_bias = struct.unpack_from("<H", data, 0x0A)[0] * 4
            hash_table = name_bias + struct.unpack_from("<H", data, 0x74)[0] * 4
            model_hashes = set()
            mesh_table = struct.unpack_from("<H", data, 0x04)[0]
            mesh_count = struct.unpack_from("<H", data, 0x20)[0]
            palette_table = name_bias + struct.unpack_from("<H", data, 0x82)[0] * 4
            for mesh_index in range(mesh_count):
                record = mesh_table + mesh_index * 0x50
                if record + 0x50 > len(data):
                    break
                flags = struct.unpack_from("<H", data, record + 0x3C)[0]
                palette_length = flags & 0xFF if flags & 0x100 else 0
                palette_start = struct.unpack_from("<H", data, record + 0x3A)[0]
                for palette_index in range(palette_start, palette_start + palette_length):
                    index_offset = palette_table + palette_index * 2
                    if index_offset + 2 > len(data):
                        break
                    joint_index = struct.unpack_from("<H", data, index_offset)[0]
                    value_offset = hash_table + joint_index * 4
                    if 0 <= value_offset and value_offset + 4 <= len(data):
                        model_hashes.add(struct.unpack_from("<I", data, value_offset)[0])
            best_bones: list[dict] = []
            best_score = -1
            for candidate_index, candidate in enumerate(unique_candidates):
                for payload in g4sk_payloads(candidate):
                    bones = g4sk_bones(payload)
                    if not bones:
                        continue
                    bone_hashes = {zlib.crc32(bone["name"].encode("ascii", errors="replace")) & 0xFFFFFFFF for bone in bones}
                    nearby_bonus = max(0, 10 - candidate_index) if candidate.parent == model_path.parent else 0
                    score = len(model_hashes & bone_hashes) * 100 + nearby_bonus
                    if score > best_score:
                        best_bones, best_score = bones, score
            return best_bones
        except (OSError, struct.error):
            return []

    def schedule_model_render(self) -> None:
        if self.model_render_after is not None:
            self.after_cancel(self.model_render_after)
        self.model_render_after = self.after(35, self.render_model_preview)

    def render_model_preview(self) -> None:
        self.model_render_after = None
        canvas = self.model_canvas
        canvas.delete("all")
        if self.preview_model is None:
            canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2, fill="#b8c2d1", text="Select a .g4md file to view its geometry")
            return
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        factor = min(1.0, 760 / width, 620 / height)
        rw, rh = max(220, int(width * factor)), max(180, int(height * factor))
        vertices = self.preview_model.vertices
        center = [sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(3)]
        radius = max(math.dist(vertex, center) for vertex in vertices) or 1.0
        cy, sy, cp, sp = math.cos(self.model_yaw), math.sin(self.model_yaw), math.cos(self.model_pitch), math.sin(self.model_pitch)
        scale = min(rw, rh) * .43 * self.model_zoom / radius
        projected = []
        for x, y, z in vertices:
            x, y, z = x-center[0], y-center[1], z-center[2]
            x, z = x*cy-z*sy, x*sy+z*cy
            y, z = y*cp-z*sp, y*sp+z*cp
            projected.append((rw/2+x*scale, rh/2-y*scale, z))
        if self.model_view_mode == "wireframe":
            from PIL import ImageDraw
            image = Image.new("RGB", (rw, rh), "#1e2530")
            draw = ImageDraw.Draw(image)
            step = max(1, len(self.preview_model.triangles)//10000)
            for a,b,c in self.preview_model.triangles[::step]:
                draw.line([projected[a][:2], projected[b][:2], projected[c][:2], projected[a][:2]], fill="#91b4d0")
            self._show_model_image(image, width, height)
            return
        frame = np.full((rh, rw, 3), (30,37,48), dtype=np.uint8)
        zbuf = np.full((rh, rw), -np.inf, dtype=np.float32)
        texture = np.asarray(self.model_texture.convert("RGB")) if self.model_view_mode == "textured" and self.model_texture else None
        step = max(1, len(self.preview_model.triangles)//(2600 if texture is not None else 7000))
        for a,b,c in self.preview_model.triangles[::step]:
            self._raster_triangle(frame, zbuf, projected[a], projected[b], projected[c], self.preview_model.uvs[a], self.preview_model.uvs[b], self.preview_model.uvs[c], texture)
        self._show_model_image(Image.fromarray(frame), width, height)

    def _show_model_image(self, image: Image.Image, width: int, height: int) -> None:
        self.model_render_image = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.BILINEAR))
        self.model_canvas.create_image(0, 0, anchor="nw", image=self.model_render_image)
        if self.model_texture_image is not None:
            self.model_canvas.create_image(12, 12, anchor="nw", image=self.model_texture_image)

    @staticmethod
    def _raster_triangle(frame: np.ndarray, zbuf: np.ndarray, a, b, c, uva, uvb, uvc, texture) -> None:
        ax,ay,az=a; bx,by,bz=b; cx,cy,cz=c
        denom=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(denom)<1e-7: return
        h,w=zbuf.shape; left=max(0,int(math.floor(min(ax,bx,cx)))); right=min(w-1,int(math.ceil(max(ax,bx,cx))))
        top=max(0,int(math.floor(min(ay,by,cy)))); bottom=min(h-1,int(math.ceil(max(ay,by,cy))))
        if left>right or top>bottom: return
        xs,ys=np.meshgrid(np.arange(left,right+1),np.arange(top,bottom+1))
        p=((by-cy)*(xs-cx)+(cx-bx)*(ys-cy))/denom; q=((cy-ay)*(xs-cx)+(ax-cx)*(ys-cy))/denom; r=1-p-q
        z=p*az+q*bz+r*cz; target=zbuf[top:bottom+1,left:right+1]; visible=(p>=0)&(q>=0)&(r>=0)&(z>target)
        if not np.any(visible): return
        target[visible]=z[visible]; pixels=frame[top:bottom+1,left:right+1]
        if texture is not None:
            u=p*uva[0]+q*uvb[0]+r*uvc[0]; v=p*uva[1]+q*uvb[1]+r*uvc[1]
            tx=np.mod(np.floor(u*texture.shape[1]).astype(int),texture.shape[1]); ty=np.mod(np.floor(v*texture.shape[0]).astype(int),texture.shape[0])
            pixels[visible]=texture[ty,tx][visible]
        else:
            normal=np.cross(np.array([bx-ax,by-ay,bz-az]),np.array([cx-ax,cy-ay,cz-az])); n=float(np.linalg.norm(normal)) or 1.0
            light=int(80+145*abs(float(normal[2])/n)); pixels[visible]=(light//2,min(235,light+25),min(255,light+50))

    def choose_folder(self) -> None:
        picked = filedialog.askdirectory(title="Choose the root folder")
        if not picked:
            return
        self.root_path = Path(picked)
        self.populate_tree()
        self.save_settings()

    def populate_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if self.root_path is None:
            return
        texture_files: list[Path] = []
        model_files: list[Path] = []
        unreadable: list[str] = []
        folders: dict[Path, str] = {self.root_path: ""}

        def report_error(error: OSError) -> None:
            if error.filename:
                unreadable.append(error.filename)

        # os.walk enumera también las carpetas vacías y acepta .G4TX/.g4tx.
        # pathlib.rglob solo dejaba visibles los directorios que tenían un archivo
        # coincidente, haciendo que una biblioteca grande pareciera incompleta.
        for current_name, directory_names, file_names in os.walk(self.root_path, onerror=report_error):
            current = Path(current_name)
            directory_names.sort(key=str.casefold)
            file_names.sort(key=str.casefold)
            # No uses dict.setdefault here: Python evalúa el valor por defecto
            # incluso cuando la carpeta ya existe, lo que creaba una copia vacía
            # de cada directorio en el nivel raíz.
            if current not in folders:
                parent_id = folders.get(current.parent, "")
                folders[current] = self.tree.insert(parent_id, "end", text=current.name, open=False)
            current_id = folders[current]
            for directory_name in directory_names:
                child = current / directory_name
                if child not in folders:
                    folders[child] = self.tree.insert(current_id, "end", text=directory_name, open=False)
            for file_name in file_names:
                file = current / file_name
                suffix = file.suffix.lower()
                if suffix == ".g4tx":
                    self.tree.insert(current_id, "end", text=file_name, values=(str(file), "g4tx"), tags=("g4tx",))
                    texture_files.append(file)
                elif self.show_models.get() and suffix in self.MODEL_EXTENSIONS:
                    self.tree.insert(
                        current_id, "end", text=f"{file_name}  [model]",
                        values=(str(file), "model"), tags=("model",),
                    )
                    model_files.append(file)

        message = f"Found {len(texture_files)} .g4tx file(s) in {len(folders) - 1} subfolder(s)."
        if self.show_models.get():
            message += f" Found {len(model_files)} 3D model(s)."
        if unreadable:
            message += f" Could not read {len(unreadable)} folder(s)."
        self.status.set(message)
        if not texture_files:
            self.heading.configure(text="No .g4tx files found")

    def open_from_tree(self, _event: tk.Event) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        values = self.tree.item(selected[0], "values")
        if values:
            path = Path(values[0])
            if len(values) > 1 and values[1] == "model":
                self.show_model_info(path)
            else:
                self.load_file(path)

    def show_model_info(self, path: Path) -> None:
        self.current_file = None
        self.textures = []
        self.selected.clear()
        for item in self.grid.winfo_children():
            item.destroy()
        self.thumbnails.clear()
        try:
            magic = path.read_bytes()[:4].decode("ascii", errors="replace")
        except OSError:
            magic = "unknown"
        nearby_textures = sorted(
            (item for item in path.parent.iterdir() if item.is_file() and item.suffix.lower() == ".g4tx"),
            key=lambda item: item.name.casefold(),
        )
        self.heading.configure(text=f"Model: {path.name}")
        self.status.set(
            f"Model detected ({magic}). G4TX textures in the same folder: {len(nearby_textures)}. "
            "Select a G4TX file to view and export its textures."
        )

    def load_file(self, path: Path) -> None:
        self.heading.configure(text=f"Loading {path.name}…")
        self.status.set("Reading container and generating previews…")
        self.update_idletasks()
        try:
            self.current_file, self.textures = path, parse_g4tx(path)
            self.selected = set(range(len(self.textures)))
            self.render_textures()
            self.heading.configure(text=f"{path.name} — {len(self.textures)} texture(s)")
            self.status.set("Select the textures you want to export.")
        except Exception as exc:
            self.heading.configure(text="Could not open file")
            self.status.set(str(exc))
            messagebox.showerror("G4TX Read Error", f"{path.name}\n\n{exc}")

    def render_textures(self) -> None:
        for item in self.grid.winfo_children():
            item.destroy()
        self.thumbnails.clear()
        for index, texture in enumerate(self.textures):
            card = ttk.Frame(self.grid, padding=5, relief="ridge")
            card.grid(row=index // 4, column=index % 4, padx=5, pady=5, sticky="nsew")
            thumb, note = self.make_thumbnail(texture)
            self.thumbnails.append(thumb)
            ttk.Label(card, image=thumb).pack()
            var = tk.BooleanVar(value=index in self.selected)
            ttk.Checkbutton(card, text=texture.name, variable=var,
                            command=lambda i=index, v=var: self.toggle(i, v.get())).pack(anchor="w", fill="x")
            ttk.Label(card, text=f"{texture.width} × {texture.height}{note}", foreground="#666").pack(anchor="w")
        for column in range(4):
            self.grid.columnconfigure(column, weight=1)

    def make_thumbnail(self, texture: Texture) -> tuple[ImageTk.PhotoImage, str]:
        try:
            if texture.payload.startswith(b"DDS "):
                payload, note = texture.payload, ""
            elif texture.payload.startswith(b"NXTCH000"):
                payload, note = nxtch_to_dds(texture.payload), " (NX)"
            else:
                raise ValueError("unknown payload format")
            image = Image.open(io.BytesIO(payload)).convert("RGBA")
            image.thumbnail((self.THUMB_SIZE, self.THUMB_SIZE), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image), note
        except Exception:
            placeholder = Image.new("RGBA", (self.THUMB_SIZE, self.THUMB_SIZE), "#444")
            return ImageTk.PhotoImage(placeholder), " (no preview)"

    def toggle(self, index: int, enabled: bool) -> None:
        (self.selected.add if enabled else self.selected.discard)(index)
        self.status.set(f"{len(self.selected)} of {len(self.textures)} texture(s) selected.")

    def set_all(self, enabled: bool) -> None:
        if not self.textures:
            return
        self.selected = set(range(len(self.textures))) if enabled else set()
        self.render_textures()
        self.status.set(f"{len(self.selected)} of {len(self.textures)} texture(s) selected.")

    def export_selected(self) -> None:
        if not self.current_file or not self.selected:
            messagebox.showinfo("Export PNG", "Select at least one texture.")
            return
        destination = filedialog.askdirectory(title="Choose the export folder")
        if not destination:
            return
        target = Path(destination) / self.current_file.stem
        target.mkdir(parents=True, exist_ok=True)
        errors, exported = [], 0
        for index in sorted(self.selected):
            texture = self.textures[index]
            try:
                payload = nxtch_to_dds(texture.payload) if texture.payload.startswith(b"NXTCH000") else texture.payload
                if not payload.startswith(b"DDS "):
                    raise ValueError("unsupported texture format")
                image = Image.open(io.BytesIO(payload)).convert("RGBA")
                image.save(target / f"{texture.name}.png", "PNG")
                exported += 1
            except Exception as exc:
                errors.append(f"{texture.name}: {exc}")
        self.status.set(f"Exported {exported} texture(s) to {target}")
        if errors:
            messagebox.showwarning("Export Complete", f"Exported: {exported}\nFailed: {len(errors)}\n\n" + "\n".join(errors[:8]))
        else:
            messagebox.showinfo("Export Complete", f"Exported {exported} texture(s) to:\n{target}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--viewer":
        model_argument = Path(sys.argv[2])
        remaining = sys.argv[3:]
        marker = remaining.index("--variant") if "--variant" in remaining else len(remaining)
        variant_paths = [Path(item) for item in remaining[:marker]]
        variant = int(remaining[marker + 1]) if marker + 1 < len(remaining) else 0
        mode_marker = remaining.index("--mode") if "--mode" in remaining else len(remaining)
        mode = remaining[mode_marker + 1] if mode_marker + 1 < len(remaining) else "solid"
        parent_marker = remaining.index("--parent") if "--parent" in remaining else len(remaining)
        parent = int(remaining[parent_marker + 1]) if parent_marker + 1 < len(remaining) else None
        port_marker = remaining.index("--control-port") if "--control-port" in remaining else len(remaining)
        control_port = int(remaining[port_marker + 1]) if port_marker + 1 < len(remaining) else None
        run_gpu_viewer(model_argument, variant_paths, max(0, min(variant, max(0, len(variant_paths) - 1))), mode, parent, control_port)
    else:
        G4TXExtractor().mainloop()
