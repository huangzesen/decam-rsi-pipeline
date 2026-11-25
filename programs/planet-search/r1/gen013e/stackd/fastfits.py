"""Fast reader for row-tiled RICE_1 (quantised float) and PLIO_1 (int mask) fpack extensions.

astropy spends most of its time in per-tile Python overhead (one tile per image row); this calls the same C codecs
in a tight loop.  The subtractive dither is not undone exactly (a constant 0.5 offset is used), which adds a
uniform +-0.5 quantum of noise (~7% of the quantum, i.e. < 0.3% of the sky noise for FZQVALUE=4).
Returns None when the extension is not of the supported kind (caller falls back to astropy)."""
import numpy as np
from astropy.io.fits.hdu.compressed._compression import decompress_rice_1_c, decompress_plio_1_c


def _row_dtype(cols):
    fields = []
    for name, fmt in zip(cols.names, cols.formats):
        f = fmt.strip()
        if "Q" in f:
            fields.append((name, ">i8", (2,)))
        elif "P" in f:
            fields.append((name, ">i4", (2,)))
        elif f.endswith("D"):
            fields.append((name, ">f8"))
        elif f.endswith("E"):
            fields.append((name, ">f4"))
        elif f.endswith("J"):
            fields.append((name, ">i4"))
        elif f.endswith("K"):
            fields.append((name, ">i8"))
        else:
            return None
    return np.dtype(fields)


def read_tiled(bt):
    h = bt.header
    try:
        ctype = h["ZCMPTYPE"].strip()
        nx, ny = int(h["ZNAXIS1"]), int(h["ZNAXIS2"])
        if int(h.get("ZTILE1", nx)) != nx or int(h.get("ZTILE2", 1)) != 1:
            return None
        rdt = _row_dtype(bt.columns)
        if rdt is None or rdt.itemsize != int(h["NAXIS1"]) or int(h["NAXIS2"]) != ny:
            return None
        tab = np.frombuffer(bt._get_raw_data(ny * rdt.itemsize, np.uint8, bt._data_offset).tobytes(), rdt)
        heap = bt._get_raw_data(int(h["PCOUNT"]), np.uint8, bt._data_offset + bt._theap)
        heap = np.asarray(heap)
        desc = tab["COMPRESSED_DATA"]
        if np.any(desc[:, 0] == 0):
            return None
        if ctype == "RICE_1":
            bs = int(h.get("ZVAL1", 32)) if str(h.get("ZNAME1", "")).strip().upper() == "BLOCKSIZE" else 32
            bp = int(h.get("ZVAL2", 4)) if str(h.get("ZNAME2", "")).strip().upper() == "BYTEPIX" else 4
            out = np.empty((ny, nx), np.int32 if bp == 4 else (np.int16 if bp == 2 else np.uint8))
            dt = f"i{bp}"
            for r in range(ny):
                n, off = desc[r]
                out[r] = np.frombuffer(decompress_rice_1_c(heap[off:off + n].tobytes(), bs, bp, nx), dt)[:nx]
            if "ZSCALE" in rdt.names:
                zs = tab["ZSCALE"].astype(np.float32)[:, None]
                zz = tab["ZZERO"].astype(np.float32)[:, None]
                img = out.astype(np.float32) * zs + zz
                if str(h.get("ZQUANTIZ", "")).strip().startswith("SUBTRACTIVE_DITHER"):
                    img += 0.0      # (I - r + 0.5) * zscale + zzero with r ~ 0.5
                zb = h.get("ZBLANK", None)
                if zb is not None:
                    img[out == int(zb)] = np.nan
                return img
            return out
        if ctype == "PLIO_1":
            out = np.empty((ny, nx), np.int32)
            for r in range(ny):
                n, off = desc[r]
                b = heap[off:off + 2 * n].view(">i2").astype("<i2").tobytes()
                out[r] = np.frombuffer(decompress_plio_1_c(b, nx), "i4")[:nx]
            return out
    except Exception:
        return None
    return None
