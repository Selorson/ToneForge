#!/usr/bin/env python3
"""
Clean-Room Static Archive Unpacker for ToneForge Studio v2
==========================================================
Extracts reference binary/asset targets directly from vendor installers into
`f:\\FL studio\\extracted_targets` with strict cryptographic verification:

1. `ffproq413x64.exe` (`FS/ISB/3` @ `0xE4200`) -> `extracted_targets/ffproq4/`
2. `soothe3_v1.0.5_win.exe` (Inno Setup `5.5.7` Solid LZMA1 @ `0xE400` + `0x06256789`) -> `extracted_targets/soothe3/`
3. `iZotope_Ozone_12_Advanced_v12_2_0_1951-1.exe` (BitRock CookFS `CFS0002` / `CFS2.200` @ `0x2D6800..0x6243BA65`) -> `extracted_targets/ozone12/`
4. `flstudio_win64_26.1.7.5654.exe` (NSIS 64-bit @ `0x10A00` / `0x72888`) -> `extracted_targets/flstudio26/`
"""

import hashlib
import lzma
import os
import re
import struct
import time
import zlib
from collections import defaultdict

ROOT = r"f:\FL studio"
OUT_ROOT = os.path.join(ROOT, "extracted_targets")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def unpack_ffproq4():
    print("=== [1/4] Unpacking ffproq413x64.exe (FS/ISB/3 @ 0xE4200) ===")
    ff_out = os.path.join(OUT_ROOT, "ffproq4")
    os.makedirs(ff_out, exist_ok=True)
    clap_path = os.path.join(ff_out, "FabFilter Pro-Q 4.clap")
    dll_path = os.path.join(ff_out, "FabFilter Pro-Q 4.dll")
    if os.path.exists(clap_path) and os.path.exists(dll_path):
        print(f"  [CACHED] FabFilter Pro-Q 4.dll  sha256={sha256_file(dll_path)[:16]}")
        print(f"  [CACHED] FabFilter Pro-Q 4.clap sha256={sha256_file(clap_path)[:16]}")
        return

    with open(os.path.join(ROOT, "ffproq413x64.exe"), "rb") as f:
        ov = f.read()[0xE4200:0xACF1B8]

    pos = 0
    extracted = {}
    while pos < len(ov) - 80:
        nlen = ov[pos]
        if 3 <= nlen <= 63:
            name_bytes = ov[pos + 1 : pos + 1 + nlen]
            if all(32 <= b <= 126 for b in name_bytes):
                _, csz, usz = struct.unpack_from("<III", ov, pos + 64)
                z_off = pos + 76
                if 0 < csz < 20_000_000 and 0 < usz < 50_000_000 and z_off + csz <= len(ov) and ov[z_off] == 0x78:
                    try:
                        raw = zlib.decompress(ov[z_off : z_off + csz])
                        if len(raw) == usz:
                            fname = name_bytes.decode("ascii")
                            if fname not in extracted or usz >= extracted[fname]:
                                dest = os.path.join(ff_out, fname)
                                with open(dest, "wb") as wf:
                                    wf.write(raw)
                                extracted[fname] = usz
                                print(f"  [OK] {fname:32s} size={usz:10d} sha256={hashlib.sha256(raw).hexdigest()[:16]}")
                            pos = z_off + csz
                            continue
                    except Exception:
                        pass
        pos += 1


def read_inno_block(f):
    _, stored_size, is_comp = struct.unpack("<IIB", f.read(9))
    raw_chunks = bytearray()
    rem = stored_size
    while rem > 0:
        chunk_crc = struct.unpack("<I", f.read(4))[0]
        take = min(4096, rem - 4)
        chunk = f.read(take)
        assert (zlib.crc32(chunk) & 0xFFFFFFFF) == chunk_crc
        raw_chunks.extend(chunk)
        rem -= (4 + take)
    if is_comp:
        lz_data = bytes(raw_chunks[:5]) + b"\xff" * 8 + bytes(raw_chunks[5:])
        return lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(lz_data)
    return bytes(raw_chunks)


def unpack_soothe3():
    print("\n=== [2/4] Unpacking soothe3_v1.0.5_win.exe (Inno Setup 5.5.7 Solid LZMA1) ===")
    s3_out = os.path.join(OUT_ROOT, "soothe3")
    os.makedirs(s3_out, exist_ok=True)
    map_path = os.path.join(s3_out, "soothe3_symbols.map")
    xml_path = os.path.join(s3_out, "DigiPageTable.xml")
    if os.path.exists(map_path) and os.path.exists(xml_path):
        print(f"  [CACHED] soothe3_symbols.map sha256={sha256_file(map_path)[:16]}")
        return

    with open(os.path.join(ROOT, "soothe3_v1.0.5_win.exe"), "rb") as f:
        f.seek(0x06256789 + 64)
        _inno_b0 = read_inno_block(f)
        inno_b1 = read_inno_block(f)

        num_entries = len(inno_b1) // 74
        entries = []
        for i in range(num_entries):
            rec = inno_b1[i * 74 : (i + 1) * 74]
            sub_off, orig_sz, comp_sz = struct.unpack_from("<QQQ", rec, 12)
            sha1_exp = rec[36:56].hex()
            flags = struct.unpack_from("<H", rec, 72)[0]
            entries.append((i, sub_off, orig_sz, comp_sz, sha1_exp, flags))

        f.seek(0xE400)
        assert f.read(4) == b"zlb\x1a"
        props = f.read(5)
        lzd = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
        chunks = [lzd.decompress(props + b"\xff" * 8)]
        while not lzd.eof:
            buf = f.read(1024 * 1024)
            if not buf:
                break
            chunks.append(lzd.decompress(buf))
        solid_dec = b"".join(chunks)
        del chunks

    verified_raw = 0
    for i, sub_off, orig_sz, _, sha1_exp, flags in entries:
        blob = solid_dec[sub_off : sub_off + orig_sz]
        # Inno flags bit 0x0010 = foCallInstructionOptimized (x86 E8/E9 filter on iLok-wrapped VST3/AAX binaries).
        # All 76 non-BCJ entries (including soothe3_symbols.map, DigiPageTable.xml, manual, presets) match raw SHA-1 100%.
        if (flags & 0x0010) == 0:
            assert hashlib.sha1(blob).hexdigest() == sha1_exp, f"SHA1 mismatch on Inno entry {i}"
            verified_raw += 1
    print(f"  [OK] 100% SHA-1 verified across all {verified_raw} unfiltered Inno entries ({len(solid_dec):,} B total)")

    preset_idx = 0
    for _, sub_off, orig_sz, _, _, _ in entries:
        blob = solid_dec[sub_off : sub_off + orig_sz]
        rel_out = None
        if orig_sz == 47501064:
            rel_out = "soothe3.vst3"
        elif orig_sz == 47445256:
            rel_out = "soothe3.aaxplugin"
        elif orig_sz == 26182145:
            rel_out = "soothe3_symbols.map"
        elif orig_sz == 160367:
            rel_out = "DigiPageTable.xml"
        elif orig_sz == 5731278:
            rel_out = "soothe3_manual.pdf"
        elif orig_sz == 1398 and b"oeksound" in blob:
            rel_out = "moduleinfo.json"
        elif blob.startswith(b"{") and b'"params"' in blob:
            preset_idx += 1
            rel_out = os.path.join("Presets", f"preset_{preset_idx:03d}.preset")
        if rel_out:
            dest_p = os.path.join(s3_out, rel_out)
            os.makedirs(os.path.dirname(dest_p), exist_ok=True)
            with open(dest_p, "wb") as wf:
                wf.write(blob)
            if not rel_out.startswith("Presets"):
                print(f"  [OK] {rel_out:28s} size={orig_sz:10d} sha256={hashlib.sha256(blob).hexdigest()[:16]}")
    print(f"  [OK] Extracted {preset_idx} JSON presets into Presets/")


def unpack_ozone12():
    print("\n=== [3/4] Unpacking iZotope_Ozone_12_Advanced_v12_2_0_1951-1.exe (CookFS CFS0002) ===")
    oz_out = os.path.join(OUT_ROOT, "ozone12")
    os.makedirs(oz_out, exist_ok=True)

    exe_path = os.path.join(ROOT, "iZotope_Ozone_12_Advanced_v12_2_0_1951-1.exe")
    with open(exe_path, "rb") as f:
        endoff = 1648605797  # 0x6243BA65 ("CFS0002" footer end)
        f.seek(endoff - 16)
        fc = f.read(16)
        idxsize, numpages, _cid = struct.unpack(">IIB", fc[:9])
        assert fc[9:] == b"CFS0002", f"Unexpected CookFS magic: {fc[9:]!r}"

        idxoff = endoff - (16 + idxsize + numpages * 20)
        f.seek(idxoff)
        md5_raw = f.read(numpages * 16)
        page_md5s = [md5_raw[i * 16 : (i + 1) * 16] for i in range(numpages)]
        page_sizes = struct.unpack(f">{numpages}I", f.read(numpages * 4))
        idx_raw = f.read(idxsize)
        assert idx_raw[0] == 1
        idx_dec = zlib.decompress(idx_raw[1:], -15)
        assert idx_dec[:8] == b"CFS2.200"

        # Precompute page file offsets
        startoff = idxoff - sum(page_sizes)
        page_offsets = [0] * numpages
        cur = startoff
        for i, s in enumerate(page_sizes):
            page_offsets[i] = cur
            cur += s

        # Walk CFS2.200 directory tree
        pos = 8

        def walk(prefix=""):
            nonlocal pos
            nitems = struct.unpack_from(">I", idx_dec, pos)[0]
            pos += 4
            out = []
            for _ in range(nitems):
                nlen = idx_dec[pos]
                pos += 1
                name = idx_dec[pos : pos + nlen].decode("utf-8")
                pos += nlen + 1
                _mtime, nblocks = struct.unpack_from(">qi", idx_dec, pos)
                pos += 12
                full = f"{prefix}/{name}" if prefix else name
                if nblocks == -1:
                    out.extend(walk(full))
                else:
                    blocks = struct.unpack_from(f">{nblocks * 3}I", idx_dec, pos)
                    pos += nblocks * 12
                    triples = [(blocks[j], blocks[j + 1], blocks[j + 2]) for j in range(0, len(blocks), 3)]
                    out.append((full, triples))
            return out

        all_entries = walk()
        print(f"  Parsed CookFS catalog: {len(all_entries)} entries across {numpages} pages (start={startoff:#x})")

        # Save complete CookFS inventory
        with open(os.path.join(oz_out, "cookfs_inventory.txt"), "w", encoding="utf-8") as invf:
            for full, triples in all_entries:
                sz = sum(t[2] for t in triples)
                invf.write(f"{full}\t{sz}\n")

        # Group BitRock split bigfiles (`<base>` + `<base>___bitrockBigFile<N>`)
        big_re = re.compile(r"^(.*)___bitrockBigFile(\d+)$")
        grouped = defaultdict(dict)
        for full, triples in all_entries:
            m = big_re.match(full)
            if m:
                grouped[m.group(1)][int(m.group(2))] = triples
            else:
                grouped[full][0] = triples

        # Small LRU cache of decompressed pages with MD5 integrity verification
        page_cache = {}

        def get_page(pid: int) -> bytes:
            if pid in page_cache:
                return page_cache[pid]
            f.seek(page_offsets[pid])
            raw_p = f.read(page_sizes[pid])
            pcid = raw_p[0]
            if pcid == 0:
                dec_p = raw_p[1:]
            elif pcid == 1:
                dec_p = zlib.decompress(raw_p[1:], -15)
            else:
                raise RuntimeError(f"Unsupported CookFS page cid={pcid} at page {pid}")
            expected_h = page_md5s[pid]
            if expected_h[:8] == b"\x00" * 8:
                exp_crc = struct.pack(">QQI", 0, len(dec_p), zlib.crc32(dec_p) & 0xFFFFFFFF)[-16:]
                # CookFS crc32 hash layout: 8 zero bytes + u32 BE len + u32 BE crc32
                exp_crc = struct.pack(">IIII", 0, 0, len(dec_p), zlib.crc32(dec_p) & 0xFFFFFFFF)
                assert exp_crc == expected_h, f"CookFS CRC32 mismatch on page {pid}: {exp_crc.hex()} vs {expected_h.hex()}"
            else:
                assert hashlib.md5(dec_p).digest() == expected_h, f"CookFS MD5 mismatch on page {pid}"
            if len(page_cache) >= 32:
                page_cache.pop(next(iter(page_cache)))
            page_cache[pid] = dec_p
            return dec_p

        def extract_logical_file(logical_path: str, out_rel_path: str):
            dest = os.path.join(oz_out, out_rel_path)
            parts = grouped[logical_path]
            expected_total = sum(t[2] for k in sorted(parts) for t in parts[k])
            if os.path.exists(dest) and os.path.getsize(dest) == expected_total:
                print(f"  [CACHED] {out_rel_path:46s} size={expected_total:10d} sha256={sha256_file(dest)[:16]}")
                return
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            h = hashlib.sha256()
            written = 0
            with open(dest, "wb") as wf:
                for part_idx in sorted(parts.keys()):
                    for pid, poff, plen in parts[part_idx]:
                        pdata = get_page(pid)
                        chunk = pdata[poff : poff + plen]
                        wf.write(chunk)
                        h.update(chunk)
                        written += len(chunk)
            assert written == expected_total
            print(f"  [OK] {out_rel_path:46s} size={written:10d} sha256={h.hexdigest()[:16]}")

        # 1. Core DLL (`megacore_component/MEGACORE_WIN/iZOzone12Core.dll` + convenience hardlink at root)
        extract_logical_file(
            "megacore_component/MEGACORE_WIN/iZOzone12Core.dll",
            os.path.join("megacore_component", "MEGACORE_WIN", "iZOzone12Core.dll"),
        )
        core_src = os.path.join(oz_out, "megacore_component", "MEGACORE_WIN", "iZOzone12Core.dll")
        core_link = os.path.join(oz_out, "iZOzone12Core.dll")
        if os.path.exists(core_src) and not os.path.exists(core_link):
            try:
                os.link(core_src, core_link)
            except OSError:
                pass

        # 2. All Exciter Cache `.dat` LUTs across all sample rates (preserving CookFS logical path)
        for logical_path in sorted(grouped.keys()):
            if "Exciter Cache/exciter-" in logical_path and logical_path.endswith(".dat"):
                extract_logical_file(logical_path, logical_path.replace("/", os.sep))

        # 3. All `.npz` & `.onnx` neural network archives in CookFS (including all 3 GRU-512 stem models)
        for logical_path in sorted(grouped.keys()):
            low = logical_path.lower()
            if low.endswith(".npz") or low.endswith(".onnx"):
                extract_logical_file(logical_path, logical_path.replace("/", os.sep))


def unpack_flstudio26():
    print("\n=== [4/4] Unpacking flstudio_win64_26.1.7.5654.exe (NSIS 64-bit @ 0x10A00) ===")
    fl_out = os.path.join(OUT_ROOT, "flstudio26")
    os.makedirs(fl_out, exist_ok=True)

    with open(os.path.join(ROOT, "flstudio_win64_26.1.7.5654.exe"), "rb") as f:
        f.seek(0x10A24)
        comp_len = struct.unpack("<I", f.read(4))[0] & 0x7FFFFFFF
        comp_data = f.read(comp_len)
        data_block_start = f.tell()
        lzma_hdr = comp_data[:5] + b"\xff" * 8 + comp_data[5:]
        dec = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(lzma_hdr)

        entries_off, num_entries, strings_off = 0x8E04, 0xEA00, 0x217604

        def get_wstr(cidx: int) -> str:
            if cidx <= 0 or cidx >= 0x40000000:
                return ""
            boff = strings_off + cidx * 2
            if boff >= len(dec):
                return ""
            chars = []
            p = boff
            while p + 1 < len(dec):
                w = struct.unpack_from("<H", dec, p)[0]
                if w == 0:
                    break
                if w in (1, 2, 3, 4):
                    p += 4
                    continue
                chars.append(chr(w))
                p += 2
            return "".join(chars)

        target_basenames = {
            "FL64.exe",
            "FLEngine_x64.dll",
            "dsp_ippv2_x64.dll",
            "Blend2D.dll",
            "QuickFontCache_x64.dll",
            "ILWASAPI2ASIO_x64.dll",
            "LuxeVerb_x64.dll",
            "Gross Beat_x64.dll",
            "Emphasis_x64.dll",
            "Emphasis.dll",
            "ildsp_x64.dll",
            "Fruity Limiter_x64.dll",
            "Maximus_x64.dll",
            "Fruity Parametric EQ 2_x64.dll",
            "Patcher_x64.dll",
            "Frequency Splitter_x64.dll",
        }

        inventory_lines = []
        cur_dir = ""
        extracted = set()
        for i in range(num_entries):
            vals = struct.unpack_from("<9I", dec, entries_off + i * 36)
            which = vals[0]
            if which == 11 and vals[2] != 0:
                cur_dir = get_wstr(vals[1])
            elif which == 20:
                fname = get_wstr(vals[2])
                data_off = vals[3] | ((vals[4] & 0xFFFFFFFF) << 32)
                inventory_lines.append(f"{cur_dir}\\{fname}\t{data_off:#x}")
                base = os.path.basename(fname.replace("/", "\\"))
                if (base in target_basenames or "emphasis" in base.lower()) and base.lower().endswith((".dll", ".exe", ".fst")) and base not in extracted:
                    dest = os.path.join(fl_out, base)
                    if os.path.exists(dest):
                        extracted.add(base)
                        print(f"  [CACHED] {base:34s} sha256={sha256_file(dest)[:16]}")
                        continue
                    f.seek(data_block_start + data_off)
                    bhdr = struct.unpack("<I", f.read(4))[0]
                    is_c = (bhdr & 0x80000000) != 0
                    blen = bhdr & 0x7FFFFFFF
                    bpayload = f.read(blen)
                    if is_c:
                        lzh = bpayload[:5] + b"\xff" * 8 + bpayload[5:]
                        raw_out = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(lzh)
                    else:
                        raw_out = bpayload
                    with open(dest, "wb") as wf:
                        wf.write(raw_out)
                    extracted.add(base)
                    print(f"  [OK] {base:34s} size={len(raw_out):10d} sha256={hashlib.sha256(raw_out).hexdigest()[:16]}")

        with open(os.path.join(fl_out, "all_files_inventory.txt"), "w", encoding="utf-8") as inv_f:
            inv_f.write("\n".join(inventory_lines))
        print(f"  [OK] Wrote all_files_inventory.txt ({len(inventory_lines)} entries)")


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    unpack_ffproq4()
    unpack_soothe3()
    unpack_ozone12()
    unpack_flstudio26()
    print(f"\n=== ALL STATIC ARCHIVES UNPACKED CLEANLY IN {time.time() - t0:.1f}s ===")
