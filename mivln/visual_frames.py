"""RGB asset bytes and host-side marker evidence, with no rendering dependency.

Frames have no filesystem path, pose, object labels or PNG metadata. The trusted
collector stamps session and observation step. Pixel overlays establish that a
marker reached the RGB input; they do NOT establish correct world-to-image
projection of a forbidden floor region, which needs separate scene validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import struct
import uuid
import zlib


def token(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("Asset/session IDs must be host-assigned 32-digit hex tokens")
    return value


def png_bytes(width, height, pixels):
    """Encode lossless 8-bit RGB PNG, without EXIF/text or third-party packages."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    scanlines = b"".join(b"\x00" + pixels[y*width*3:(y+1)*width*3] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def validate_png(data, width, height):
    """Validate our canonical RGB encoding, dimensions, CRCs and bounded data.

    The collector starts with raw RGB and strips all ancillary metadata. An
    imported PNG must go through that collector rather than smuggling labels or
    paths in text/EXIF chunks. This verifier is for our three-chunk encoding.
    """
    if (type(data) is not bytes or type(width) is not int or type(height) is not int
            or not 0 < width <= 4096 or not 0 < height <= 4096 or data[:8] != b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not a supported bounded RGB PNG")
    chunks, offset = [], 8
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("Truncated PNG chunk")
        size = struct.unpack(">I", data[offset:offset+4])[0]
        end = offset + 12 + size
        if end > len(data):
            raise ValueError("Truncated PNG data")
        kind, payload = data[offset+4:offset+8], data[offset+8:offset+8+size]
        crc = struct.unpack(">I", data[end-4:end])[0]
        if crc != zlib.crc32(kind + payload) & 0xffffffff:
            raise ValueError("PNG CRC mismatch")
        chunks.append((kind, payload))
        offset = end
    if [kind for kind, _ in chunks] != [b"IHDR", b"IDAT", b"IEND"] or chunks[-1][1]:
        raise ValueError("Expected metadata-free canonical RGB PNG")
    if chunks[0][1] != struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0):
        raise ValueError("PNG header differs from the public image resource")
    expected = height * (1 + width * 3)
    try:
        decoder = zlib.decompressobj()
        rows = decoder.decompress(chunks[1][1], expected + 1)
    except zlib.error as error:
        raise ValueError("Invalid PNG compression") from error
    if len(rows) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Wrong PNG scanline size or trailing compressed data")
    stride = 1 + width * 3
    if any(rows[y*stride] != 0 for y in range(height)):
        raise ValueError("Unsupported PNG scanline filter")
    return b"".join(rows[y*stride+1:(y+1)*stride] for y in range(height))


@dataclass(frozen=True)
class RGBFrame:
    """An immutable collected observation; step t is BEFORE action t."""
    session_id: str
    step: int
    width: int
    height: int
    pixels: bytes
    asset_id: str

    @classmethod
    def collect(cls, session_id, step, width, height, pixels, asset_id=None):
        token(session_id)
        if type(step) is not int or step < 0:
            raise ValueError("Observation step must be nonnegative")
        if (type(width) is not int or type(height) is not int or not 0 < width <= 4096
                or not 0 < height <= 4096 or type(pixels) is not bytes or len(pixels) != width * height * 3):
            raise ValueError("Expected bounded width/height and exactly width*height*3 RGB bytes")
        return cls(session_id, step, width, height, pixels, token(asset_id if asset_id is not None else uuid.uuid4().hex))

    def validate(self):
        # Direct dataclass construction also has to obey the collector contract.
        self.collect(self.session_id, self.step, self.width, self.height, self.pixels, self.asset_id)

    def png(self):
        self.validate()
        return png_bytes(self.width, self.height, self.pixels)

    def resource(self):
        return {"asset_id": self.asset_id, "sha256": hashlib.sha256(self.png()).hexdigest(),
                "width": self.width, "height": self.height, "media_type": "image/png"}


@dataclass(frozen=True)
class MarkerEvidence:
    """Evaluator-only proof of exact overlay bytes, not world geometry proof."""
    region_id: str
    original_sha256: str
    marked_sha256: str
    pixel_indices: tuple
    color_rgb: tuple


def mark_frame(frame, region_id, pixel_indices, color_rgb=(255, 0, 0), asset_id=None):
    """Overlay a host-projected mask into the ACTUAL policy RGB asset.

    A future scene adapter must obtain this mask from the frozen floor geometry.
    Arbitrary image rectangles are suitable only for synthetic contract demos.
    Original category images and private diagnostic masks are never inputs here.
    """
    frame.validate()
    if type(region_id) is not str or not region_id.strip():
        raise ValueError("Marker must bind an evaluator-owned region ID")
    indices = tuple(pixel_indices)
    color = tuple(color_rgb)
    if (not indices or any(type(i) is not int or not 0 <= i < frame.width * frame.height for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Marker mask must contain unique, in-range pixel indices")
    if len(color) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in color):
        raise ValueError("Marker color must have three byte components")
    pixels = bytearray(frame.pixels)
    for index in indices:
        pixels[3*index:3*index+3] = bytes(color)
    marked = RGBFrame.collect(frame.session_id, frame.step, frame.width, frame.height, bytes(pixels), asset_id)
    if marked.asset_id == frame.asset_id:
        raise ValueError("A modified RGB asset needs a new identity")
    proof = MarkerEvidence(region_id, frame.resource()["sha256"], marked.resource()["sha256"], indices, color)
    return marked, proof


def verify_marker(original, marked, evidence, region_id, min_pixels):
    """Reject relabelled, invisible, cross-step or mismatched overlay evidence."""
    original.validate()
    marked.validate()
    if (type(min_pixels) is not int or min_pixels < 1 or type(evidence) is not MarkerEvidence
            or evidence.region_id != region_id
            or (original.session_id, original.step, original.width, original.height)
               != (marked.session_id, marked.step, marked.width, marked.height)):
        raise ValueError("Marker evidence has incorrect identity, step, dimensions or threshold")
    expected, _ = mark_frame(original, region_id, evidence.pixel_indices, evidence.color_rgb)
    if (evidence.original_sha256 != original.resource()["sha256"]
            or evidence.marked_sha256 != marked.resource()["sha256"] or expected.pixels != marked.pixels):
        raise ValueError("Marker evidence does not bind the actual RGB bytes")
    changed = sum(original.pixels[3*i:3*i+3] != marked.pixels[3*i:3*i+3] for i in evidence.pixel_indices)
    if changed < min_pixels:
        raise ValueError("Insufficient visibly changed marker pixels at delivery")
    return changed
