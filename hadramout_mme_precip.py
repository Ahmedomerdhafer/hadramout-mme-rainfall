#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
  نظام التنبؤ الهيدرولوجي التجميعي متعدد النماذج (MME) — محافظة حضرموت وخليج عدن
  Multi-Model Super-Ensemble 24h Accumulated Rainfall — Hadramout, Yemen
==============================================================================

  النماذج والأوزان            : GFS(الأمريكي) 40% | ECMWF(الأوروبي) 30% | GEM(الكندي) 20% | ICON(الألماني) 10%
  الفترة                      : 00Z  ->  +24h  (تراكم 24 ساعة)
  مصادر البيانات (مفتوحة بالكامل، بدون مفاتيح API):
      * ECMWF  : https://data.ecmwf.int/forecasts/   (GRIB2 0.25° — استخراج جزئي عبر HTTP Range)
      * GFS    : NOAA Big Data Program على AWS/GCP   (GRIB2 0.25° — سجل APCP عبر HTTP Range)
      * ICON   : DWD Open Data                       (GRIB2 شبكة عشوائية — إعادة استيفاء للشبكة المنتظمة)
      * GEM    : Open-Meteo API                      (نموذج GEM الكندي، شبكة نقاط)
  احتياطي (Fallback)          : Open-Meteo لأي نموذج يفشل تحميله، ثم نمط Offline تخليقي.

  المخرجات:
      1) hadramout_mme_precip_24h.png        — الخريطة الكارتوجرافية النهائية (300 dpi)
      2) hadramout_mme_stations_24h.csv      — قيم التراكم عند المحطات الهيدرولوجية
      3) mme_report.txt                      — سجل التشغيل وميتاداتا كل نموذج

  التشغيل:  python3 hadramout_mme_precip.py
            python3 hadramout_mme_precip.py --offline     (بدون إنترنت: بيانات تخليقية للاختبار)
==============================================================================
"""

import argparse
import bz2
import io
import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as _fm

# تسجيل خط عربي (Tajawal) لعرض نصوص عربية سليمة
import os as _os
_FONT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets", "fonts")
for _v in ("Regular", "Medium", "Bold"):
    _fp = _os.path.join(_FONT_DIR, f"Tajawal-{_v}.ttf")
    if _os.path.exists(_fp):
        _fm.fontManager.addfont(_fp)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

try:
    from arabic_reshaper import reshape as _reshape
    from bidi.algorithm import get_display as _bidi
    _HAS_AR = True
except Exception:
    _HAS_AR = False


def ar(text):
    """تشكيل نص عربي للعرض الصحيح (اتصال الحروف + اتجاه من اليمين لليسار)."""
    t = str(text)
    if not _HAS_AR:
        return t
    return _bidi(_reshape(t))


AR_MODEL = {"GFS": "الأمريكي", "ECMWF": "الأوروبي", "GEM": "الكندي", "ICON": "الألماني"}


def LTR(text):
    """تطويق مقطع لاتيني/رقمي (تواريخ، نوافذ، نسب) ليبقى ترتيبه يسار→يمين داخل نص عربي."""
    return "\u202A" + str(text) + "\u202C"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Downloading.*")

try:
    import eccodes  # واجهة GRIB الرسمية (ECMWF ecCodes)
except Exception:  # pragma: no cover
    eccodes = None


# ==============================================================================
# 1. الإعدادات العامة والمجال الجغرافي لشبكة التصدير
# ==============================================================================
WEIGHTS = {
    "GFS":   0.40,   # الأمريكي — NOAA GFS 0.25°
    "ECMWF": 0.30,   # الأوروبي — IFS HRES 0.25°
    "GEM":   0.20,   # الكندي  — MSC GEM 0.25°
    "ICON":  0.10,   # الألماني — DWD ICON ~13km
}

# مجال الخريطة (حضرموت + خليج عدن)
LON_MIN, LON_MAX = 42.0, 56.0
LAT_MIN, LAT_MAX = 10.0, 20.0

# شبكة التصدير الموحدة (0.10° ≈ 11 كم) — تُبنى عليها جميع الحقول قبل الدمج
TARGET_RES = 0.10
grid_lats = np.round(np.arange(LAT_MAX, LAT_MIN - 1e-9, -TARGET_RES), 4)   # من الشمال للجنوب
grid_lons = np.round(np.arange(LON_MIN, LON_MAX + 1e-9,  TARGET_RES), 4)   # من الغرب للشرق
GLON, GLAT = np.meshgrid(grid_lons, grid_lats)                             # الشكل: (nlat, nlon)

# هوامش أمان عند الاستيفاء من الشبكات الأصلية
PAD = 3.0

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE_DIR, "data_cache")
OUT_DIR    = os.path.join(BASE_DIR, "outputs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

UA = {"User-Agent": "HadramoutMME/1.0 (hydrological-forecast research script)"}


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}", flush=True)


# ==============================================================================
# 2. أدوات الشبكات: تحميل متين + فك ترميز GRIB2 + إعادة الاستيفاء
# ==============================================================================
def http_get(url, headers=None, timeout=300, tries=3, sleep=2):
    """تحميل مع إعادة محاولة تلقائية (يرد 200 أو 206)."""
    last = None
    h = dict(UA)
    if headers:
        h.update(headers)
    for i in range(tries):
        try:
            r = requests.get(url, headers=h, timeout=timeout)
            if r.status_code in (200, 206):
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(sleep)
    raise RuntimeError(f"تعذّر التحميل {url} ({last})")


def http_head_ok(url, timeout=45):
    try:
        r = requests.head(url, headers=UA, timeout=timeout, allow_redirects=True)
        return r.status_code == 200, int(r.headers.get("Content-Length", 0) or 0)
    except Exception:
        return False, 0


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


def grib_handle(data):
    """فك ترميز رسالة GRIB من بايتات (ecCodes يتطلب ملفاً حقيقياً)."""
    if eccodes is None:
        raise RuntimeError("مكتبة eccodes غير مثبتة — pip install eccodes")
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".grib2")
    try:
        os.write(fd, data)
        os.close(fd)
        with open(path, "rb") as fh:
            msg = eccodes.codes_grib_new_from_file(fh)
        if msg is None:
            raise RuntimeError("لا توجد رسالة GRIB صالحة")
        return msg
    finally:
        if os.path.exists(path):
            os.remove(path)


def grib_field(msg):
    """استخراج القيم وشبكة الإحداثيات من رسالة GRIB (منتظمة أو عشوائية)."""
    vals = np.asarray(eccodes.codes_get_array(msg, "values"), dtype=np.float64)
    miss = eccodes.codes_get(msg, "missingValue")
    if eccodes.codes_get(msg, "bitmapPresent"):
        vals[vals == miss] = np.nan
    grid_type = eccodes.codes_get(msg, "gridType")

    if grid_type in ("regular_ll", "rotated_ll"):
        ni = eccodes.codes_get(msg, "Ni")
        nj = eccodes.codes_get(msg, "Nj")
        la1 = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
        la2 = eccodes.codes_get(msg, "latitudeOfLastGridPointInDegrees")
        lo1 = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")
        lo2 = eccodes.codes_get(msg, "longitudeOfLastGridPointInDegrees")
        lats = np.linspace(la1, la2, nj)
        field = vals.reshape(nj, ni)

        # بناء محاور الطول بشكل متين: بعض النماذج تبدأ عند 180° أو 0° (نظام 0..360)
        dx = eccodes.codes_get(msg, "iDirectionIncrementInDegrees")
        if dx and abs(((lo2 - lo1) % 360.0) - (ni - 1) * dx) < 0.05:
            lons = lo1 + dx * np.arange(ni)
        else:
            lons = np.linspace(lo1, lo2, ni)
        wrap360 = (lons.max() - lons.min()) > 200.0
        lons = np.where(lons > 180.0, lons - 360.0, lons)

        if lats[0] < lats[-1]:                 # ضمان ترتيب تنازلي لخطوط العرض
            lats = lats[::-1]
            field = field[::-1, :]
        if lons[0] > lons[-1]:                 # إعادة ترتيب الطول تصاعدياً
            order = np.argsort(lons)
            lons = lons[order]
            field = field[:, order]
        if wrap360 and lons[0] > LON_MIN - PAD - 1.0:   # تمديد الغلاف العالمي 360°
            lons = np.concatenate([lons - 360.0, lons])
            field = np.concatenate([field, field], axis=1)
        return dict(kind="regular", lats=lats, lons=lons, values=field)

    # شبكة عشوائية (ICON Icosahedral) — إحداثيات كل خلية داخل الرسالة
    clat = np.asarray(eccodes.codes_get_array(msg, "latitudes"), dtype=np.float64)
    clon = np.asarray(eccodes.codes_get_array(msg, "longitudes"), dtype=np.float64)
    return dict(kind="unstructured", lats=clat, lons=clon, values=vals)


def regrid(src, out_lats=grid_lats, out_lons=grid_lons):
    """إعادة استيفاء أي حقل (منتظم/عشوائي) إلى شبكة التصدير الموحدة (nlat, nlon)."""
    if src["kind"] == "regular":
        # قص النطاق المطلوب (+هامش) لتسريع الاستيفاء
        la, lo, v = src["lats"], src["lons"], src["values"]
        m_lat = (la >= out_lats.min() - PAD) & (la <= out_lats.max() + PAD)
        m_lon = (lo >= out_lons.min() - PAD) & (lo <= out_lons.max() + PAD)
        la, lo, v = la[m_lat], lo[m_lon], v[np.ix_(m_lat, m_lon)]
        vv = np.nan_to_num(v, nan=0.0)
        interp = RegularGridInterpolator(
            (la, lo), vv, method="linear", bounds_error=False, fill_value=0.0
        )
        pts = np.column_stack([GLAT.ravel(), GLON.ravel()])
        return interp(pts).reshape(GLAT.shape)

    # شبكة عشوائية: أقرب 8 خلايا بترجيح المسافة العكسية (IDW) — يحفظ تدرج الأمطار التضاريسي
    la, lo, v = src["lats"], src["lons"], np.nan_to_num(src["values"], nan=0.0)
    m = ((la > out_lats.min() - PAD) & (la < out_lats.max() + PAD) &
         (lo > out_lons.min() - PAD) & (lo < out_lons.max() + PAD))
    la, lo, v = la[m], lo[m], v[m]
    tree = cKDTree(np.column_stack([la, lo]))
    pts = np.column_stack([GLAT.ravel(), GLON.ravel()])
    dist, idx = tree.query(pts, k=8)
    w = 1.0 / np.maximum(dist, 1e-6) ** 2
    w /= w.sum(axis=1, keepdims=True)
    return (v[idx] * w).sum(axis=1).reshape(GLAT.shape)


# ==============================================================================
# 3. اكتشاف أحدث دورة تشغيل متاحة لكل نموذج (+24h على الأقل)
# ==============================================================================
def utc_now():
    return datetime.now(timezone.utc)


def candidate_runs():
    """قوائم دورات التشغيل المرشحة (الأحدث أولاً) للأيام الثلاثة الماضية."""
    now = utc_now()
    days = [(now - timedelta(days=d)).strftime("%Y%m%d") for d in range(0, 3)]
    runs = []
    for d in days:
        for hh in ("12", "06", "00"):
            runs.append((d, hh))
    return runs


def discover_ecmwf():
    """أحدث دورة IFS متاحة تحتوي ملف -24h (open data)."""
    root = "https://data.ecmwf.int/forecasts/"
    html = http_get(root, timeout=60).text
    dates = sorted(set(re.findall(r'/forecasts/(\d{8})/', html)), reverse=True)
    now_day = utc_now().strftime("%Y%m%d")
    dates = [d for d in dates if d <= now_day][:3]
    for d in dates:
        for hh in ("12", "06", "00"):
            base = f"{root}{d}/{hh}z/ifs/0p25/oper/{d}{hh}0000-24h-oper-fc.grib2"
            ok, size = http_head_ok(base)
            # إن حُذف الملف من الخادم بعدما خزّنّا مقطع tp منه، نعتبر الدورة متاحة
            cached = os.path.exists(cache_path(f"ecmwf_{d}{hh}_tp_0-24.grib2"))
            if (ok and size > 1_000_000) or cached:
                return d, hh
    raise RuntimeError("لم يتم العثور على دورة ECMWF متاحة")


def discover_gfs():
    """أحدث دورة GFS 0.25° على AWS/GCP تحتوي سجل APCP (0-1 day acc)."""
    mirrors = [
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f024",
        "https://storage.googleapis.com/global-forecast-system/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f024",
    ]
    for d, h in candidate_runs():
        for mir in mirrors:
            idx_url = mir.format(d=d, h=h) + ".idx"
            try:
                txt = http_get(idx_url, timeout=60).text
            except Exception:
                continue
            if not txt.lstrip()[:1].isdigit():
                continue
            if gfs_apcp_record(txt):
                return d, h, mir.format(d=d, h=h)
    raise RuntimeError("لم يتم العثور على دورة GFS متاحة")


def gfs_apcp_record(idx_text):
    """قراءة فهرس NCEP وإرجاع (byte-range) لسجل APCP التراكمي 0-24h."""
    lines = [l for l in idx_text.splitlines() if l.strip()]
    parsed = []
    for l in lines:
        p = l.split(":")
        try:
            off = int(p[1])
        except (ValueError, IndexError):
            continue
        parsed.append((off, p[3] if len(p) > 3 else "", p[4] if len(p) > 4 else "",
                       p[5] if len(p) > 5 else ""))
    # المطلوب: تراكم 0-1 يوم، وإلا مجموع التراكمات الساعية من 1..24
    for i, (off, name, level, rng) in enumerate(parsed):
        if name == "APCP" and level == "surface" and "0-1 day" in rng:
            end = parsed[i + 1][0] - 1 if i + 1 < len(parsed) else None
            return [("0-24h (0-1 day acc)", off, end)]
    return None


def discover_icon():
    """أحدث دورة ICON-global تحتوي TOT_PREC عند +24h (DWD Open Data)."""
    root = "https://opendata.dwd.de/weather/nwp/icon/grib/"
    for d, h in candidate_runs():
        url = (f"{root}{h}/tot_prec/icon_global_icosahedral_single-level_"
               f"{d}{h}_024_TOT_PREC.grib2.bz2")
        ok, size = http_head_ok(url)
        if ok and size > 100_000:
            return d, h, url
    raise RuntimeError("لم يتم العثور على دورة ICON متاحة")



# ==============================================================================
# 3ب) طبقة Open-Meteo (دفعات + احترام حد المعدل) — تُستخدم لـ GEM وأيضاً احتياطياً
# ==============================================================================
OM_URL = "https://api.open-meteo.com/v1/forecast"
OM_LOCK = __import__("threading").Lock()
OM_MIN_GAP = 2.0            # أقل فاصلة زمنية (ثانية) بين طلبين متتاليين
OM_LAST_CALL = [0.0]
import threading as _th
OM_MODELS = {"ECMWF": ["ecmwf_ifs025"],
             "GFS":   ["gfs_seamless"],
             "ICON":  ["icon_seamless"],
             "GEM":   ["gem_seamless"]}


def reference_day():
    """اليوم المرجعي للنافذة 00Z→+24h: آخر منتصف ليل UTC مرّ قبل لحظة التشغيل،
    حتى تبقى النافذة مستقبلية بالكامل في ساعات الليل الأولى."""
    return utc_now().strftime("%Y-%m-%d")


def _om_window_indices(times, vals, target=None):
    """تحديد أحدث نافذة 00Z→+24h مكتملة وإرجاع (مؤشراتها، اسم اليوم)."""
    from collections import defaultdict
    hours = defaultdict(set)
    for t in times:
        hours[t[:10]].add(t[11:])
    complete = [d for d, hs in hours.items() if len(hs - {"00:00"}) >= 22]
    if not complete:
        complete = sorted(hours)
    # الأولوية لليوم المرجعي المطلوب توحيد النافذة عليه (مطابقة ECMWF/GFS)
    day0 = target if (target in complete or target in hours) else max(complete)
    nxt = (datetime.strptime(day0, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    idx = [i for i, t in enumerate(times)
           if (t[:10] == day0 and t[11:] != "00:00") or t == f"{nxt}T00:00"]
    return idx, day0


def openmeteo_gridded(model_key, step=0.5, chunk=300, max_wait=200, target_day=None):
    """
    جلب حقل تراكم 24h من Open-Meteo على شبكة نقاط منتظمة (00Z→+24h).
    يقسّم الطلبات إلى دفعات صغيرة لتفادي 414 URI Too Large و429 Rate Limit.
    """
    la = np.round(np.arange(LAT_MIN - 1, LAT_MAX + 1 + 1e-9, step), 2)
    lo = np.round(np.arange(LON_MIN - 1, LON_MAX + 1 + 1e-9, step), 2)
    pts = [(a, b) for a in la for b in lo]
    target_day = target_day or reference_day()
    t0 = datetime.strptime(target_day, "%Y-%m-%d")
    day_start = target_day
    day_end = (t0 + timedelta(days=1)).strftime("%Y-%m-%d")

    cache_key = cache_path(f"om_{model_key}_{target_day}.npz")
    if os.path.exists(cache_key):
        z = np.load(cache_key)
        src = dict(kind="regular", lats=z["lats"], lons=z["lons"], values=z["values"])
        log(f"   {model_key} via Open-Meteo — من الذاكرة المؤقتة ({target_day}) "
            f"شبكة {z['values'].shape}  أقصى قيمة={z['values'].max():.1f} ملم")
        return dict(src=src, meta=dict(shortName="tp", stepRange="0-24", units="mm",
                                       run=f"Open-Meteo/{z['model']}",
                                       window=f"{target_day} 00Z→+24h"))

    names = OM_MODELS.get(model_key, ["best_match"])
    results, used_model, err = None, None, None
    for name in names:
        out, ok = [], True
        for i in range(0, len(pts), chunk):
            ch = pts[i:i + chunk]
            params = {
                "latitude": ",".join(f"{a:.2f}" for a, _ in ch),
                "longitude": ",".join(f"{b:.2f}" for _, b in ch),
                "hourly": "precipitation", "models": name,
                "start_date": day_start, "end_date": day_end, "timezone": "GMT",
            }
            waited, res = 0, None
            while True:
                try:
                    with OM_LOCK:
                        gap = time.time() - OM_LAST_CALL[0]
                        if gap < OM_MIN_GAP:
                            time.sleep(OM_MIN_GAP - gap)
                        OM_LAST_CALL[0] = time.time()
                    r = requests.get(OM_URL, params=params, headers=UA, timeout=180)
                except Exception as e:
                    err = f"{type(e).__name__}"; time.sleep(3); waited += 3
                    if waited > max_wait: ok = False; break
                    continue
                if r.status_code == 200:
                    res = r.json(); break
                if r.status_code in (414, 429, 500, 502, 503):
                    pause = 62 if r.status_code == 429 else 4   # حد الدقيقة يتطلب صبراً
                    if waited + pause > max_wait:
                        err = f"HTTP {r.status_code}"; ok = False; break
                    time.sleep(pause); waited += pause
                    continue
                err = f"HTTP {r.status_code} {r.text[:90]}"; ok = False; break
            if not ok: break
            out.extend(res if isinstance(res, list) else [res])
            time.sleep(0.4)                       # تلطيف معدل الطلبات
        if ok and out:
            results, used_model = out, name
            break
    if not results:
        raise RuntimeError(f"Open-Meteo ({model_key}) فشل: {err}")

    acc = np.zeros((len(la), len(lo)), dtype=np.float64)
    day0 = None
    for k, item in enumerate(results):
        i, j = divmod(k, len(lo))
        times = item["hourly"]["time"]
        vals = np.nan_to_num(np.asarray(item["hourly"]["precipitation"], dtype=np.float64), nan=0.0)
        idx, d0 = _om_window_indices(times, vals, target=target_day)
        if day0 is None: day0 = d0
        acc[i, j] = vals[idx].sum()

    src = dict(kind="regular", lats=la[::-1], lons=lo, values=acc[::-1, :])
    meta = dict(shortName="tp", stepRange="0-24", units="mm",
                run=f"Open-Meteo/{used_model}", window=f"{day0} 00Z→+24h")
    np.savez_compressed(cache_key, values=acc[::-1, :], lats=la[::-1], lons=lo,
                        model=used_model, day=day0)
    log(f"   {model_key} via Open-Meteo[{used_model}]  نقاط={len(pts)}  "
        f"نافذة={day0} 00Z→+24h  شبكة {acc.shape}  أقصى قيمة={acc.max():.1f} ملم")
    return dict(src=src, meta=meta)


def shorten(text, n=170):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n] + " …"


# ==============================================================================
# 4. جلب حقل الأمطار التراكمية 24h لكل نموذج
# ==============================================================================
def fetch_ecmwf_precipitation_24h(run=None):
    run = run or discover_ecmwf()
    d, h = run[0], run[1]
    base = f"https://data.ecmwf.int/forecasts/{d}/{h}z/ifs/0p25/oper/{d}{h}0000-24h-oper-fc.grib2"
    key = cache_path(f"ecmwf_{d}{h}_tp_0-24.grib2")
    # الأولوية للكاش المحلي (قد يحذف ECMWF ملفات الدورة القديمة من الخادم)
    if os.path.exists(key) and os.path.getsize(key) > 100_000:
        raw = open(key, "rb").read()
    else:
        idx_txt = http_get(base.replace(".grib2", ".index"), timeout=120).text
        rec = None
        for line in idx_txt.splitlines():
            if '"param": "tp"' in line or '"param":"tp"' in line:
                if '"step": "24"' in line or '"step":"24"' in line:
                    o = int(re.search(r'"_offset":\s*(\d+)', line).group(1))
                    ln = int(re.search(r'"_length":\s*(\d+)', line).group(1))
                    rec = (o, ln)
                    break
        if rec is None:
            raise RuntimeError("سجل tp غير موجود في فهرس ECMWF")
        off, length = rec
        r = http_get(base, headers={"Range": f"bytes={off}-{off + length - 1}"}, timeout=300)
        raw = r.content if r.status_code == 206 else r.content[off:off + length]
        if len(raw) != length:
            raise RuntimeError(f"حجم شريحة ECMWF غير صحيح ({len(raw)} != {length})")
        open(key, "wb").write(raw)
    msg = grib_handle(raw)
    msg0 = grib_handle(raw)
    meta = dict(shortName=eccodes.codes_get(msg0, "shortName"),
                stepRange=eccodes.codes_get(msg0, "stepRange"),
                units=eccodes.codes_get(msg0, "units"),
                run=f"{d}/{h}Z",
                window=f"{d[4:6]}-{d[6:]} {h}Z→+24h")
    eccodes.codes_release(msg0)
    src = grib_field(grib_handle(raw))
    log(f"   ECMWF IFS  run={d}/{h}Z  step={meta['stepRange']}  "
        f"{len(raw)/1024:.0f} KB  شبكة أصلية {src['values'].shape}")
    return dict(src=src, meta=meta)


def fetch_gfs_precipitation_24h(run=None):
    if run is None:
        d, h, base = discover_gfs()
    else:
        d, h, base = run[0], run[1], run[2]
    idx_txt = http_get(base + ".idx", timeout=60).text
    if not idx_txt.lstrip()[:1].isdigit():
        idx_txt = http_get(base + ".idx", timeout=60).text
    recs = gfs_apcp_record(idx_txt)
    if not recs:
        raise RuntimeError("سجل APCP 0-1 day غير موجود في GFS")
    _, off, end = recs[0]
    key = cache_path(f"gfs_{d}{h}_apcp_0-24.grib2")
    if os.path.exists(key) and os.path.getsize(key) > 10_000:
        raw = open(key, "rb").read()
    else:
        hdr = {"Range": f"bytes={off}-{end}"} if end else {"Range": f"bytes={off}-"}
        r = http_get(base, headers=hdr, timeout=600)
        raw = r.content
        open(key, "wb").write(raw)
    msg = grib_handle(raw)
    meta = dict(shortName=eccodes.codes_get(msg, "shortName"),
                stepRange=eccodes.codes_get(msg, "stepRange"),
                units=eccodes.codes_get(msg, "units"),
                run=f"{d}/{h}Z",
                window=f"{d[4:6]}-{d[6:]} {h}Z→+24h")
    src = grib_field(msg)
    eccodes.codes_release(msg)
    log(f"   GFS 0.25°  run={d}/{h}Z  step={meta['stepRange']}  "
        f"{len(raw)/1024:.0f} KB  شبكة أصلية {src['values'].shape}")
    return dict(src=src, meta=meta)


def _icon_has_coords(raw):
    """هل تحمل رسالة ICON إحداثيات الخلايا داخلها؟ (بعض الإصدارات لا تحملها)"""
    try:
        msg = grib_handle(raw)
        try:
            eccodes.codes_get_array(msg, "latitudes")
            return msg
        except Exception:
            eccodes.codes_release(msg)
            return None
    except Exception:
        return None


def fetch_icon_precipitation_24h(run=None):
    """
    ICON: الأولوية لملف TOT_PREC الأصلي من DWD Open Data (شبكة عشوائية ~13 كم).
    إن لم تحمل الرسالة إحداثيات الخلايا (لا يمكن إسنادها مكانياً) ⇒ Open-Meteo[icon].
    """
    if run is None:
        d, h, url = discover_icon()
    else:
        d, h, url = run[0], run[1], run[2]
    key = cache_path(f"icon_{d}{h}_totprec_024.grib2")
    if os.path.exists(key) and os.path.getsize(key) > 100_000:
        raw = open(key, "rb").read()
    else:
        comp = http_get(url, timeout=600).content
        raw = bz2.decompress(comp) if url.endswith(".bz2") else comp
        open(key, "wb").write(raw)

    msg = _icon_has_coords(raw)
    if msg is None:
        log(f"   ⚠ ملف ICON الأصلي ({d}/{h}Z, {len(raw)/1024/1024:.1f} MB) بلا إحداثيات خلايا "
            f"⇒ استخدام Open-Meteo[icon] لنفس النموذج")
        return openmeteo_gridded("ICON", step=0.5)

    meta = dict(shortName=eccodes.codes_get(msg, "shortName"),
                stepRange=eccodes.codes_get(msg, "stepRange"),
                units=eccodes.codes_get(msg, "units"),
                run=f"{d}/{h}Z (DWD native)",
                window=f"{d[4:6]}-{d[6:]} {h}Z→+24h")
    src = grib_field(msg)
    eccodes.codes_release(msg)
    log(f"   ICON global  run={d}/{h}Z  step={meta['stepRange']}  "
        f"{len(raw)/1024/1024:.1f} MB  خلايا={src['values'].size}")
    return dict(src=src, meta=meta)


def fetch_gem_precipitation_24h(run=None):
    """GEM (كندا): لا يتوفر GRIB مفتوح مباشر ⇒ يُجلب عبر Open-Meteo على شبكة نقاط."""
    return openmeteo_gridded("GEM", step=0.5)


# ---------- احتياطي عام: Open-Meteo لأي نموذج ----------
def fetch_openmeteo_fallback(model_key):
    """احتياطي عام: أي نموذج عبر Open-Meteo."""
    return openmeteo_gridded(model_key, step=0.5, chunk=300)


def fetch_synthetic_fallback(model_key):
    """نمط Offline: حقل تخليقي واقعي (خلية حمل حراري + تأثير تضاريسي) لاختبار المسار الكامل."""
    rng = np.random.default_rng(abs(hash(model_key)) % 2**31)
    field = np.zeros_like(GLAT)
    for _ in range(3):
        c_lon = rng.uniform(LON_MIN + 2, LON_MAX - 2)
        c_lat = rng.uniform(LAT_MIN + 2, LAT_MAX - 2)
        amp = rng.uniform(5, 45)
        sx, sy = rng.uniform(1.2, 2.8), rng.uniform(0.9, 2.0)
        field += amp * np.exp(-(((GLON - c_lon) / sx) ** 2 + ((GLAT - c_lat) / sy) ** 2))
    field *= (1.0 + 0.35 * np.sin(np.radians((GLON - LON_MIN) * 22))
              * np.cos(np.radians((GLAT - LAT_MIN) * 18)))
    field = np.maximum(field, 0.0) + rng.normal(0, 0.05, GLAT.shape).clip(-0.05, 0.05)
    log(f"   ⚠ نمط Offline تخليقي ({model_key}) — لا يُستخدم للتنبؤ التشغيلي")
    return dict(src=dict(kind="regular", lats=grid_lats, lons=grid_lons, values=field),
                meta=dict(shortName="tp", stepRange="0-24", units="mm", run=f"SYNTHETIC/{model_key}"))


FETCHERS = {
    "ECMWF": fetch_ecmwf_precipitation_24h,
    "GFS":   fetch_gfs_precipitation_24h,
    "ICON":  fetch_icon_precipitation_24h,
    "GEM":   fetch_gem_precipitation_24h,
}


def to_mm(res):
    """توحيد الوحدات إلى ملم: ECMWF IFS يعطي tp بالمتر، وICON/GFS يعطيانها بـ kg/m² (= ملم)."""
    src, units = res["src"], res["meta"].get("units", "")
    v = np.asarray(src["values"], dtype=np.float64)
    if "m" == units.strip() or units.strip().endswith("m**-1") or units.strip() == "m":
        v = v * 1000.0
        res["meta"]["units_converted"] = "m -> mm"
    elif v.size and np.nanmax(v) > 0:
        peak = np.nanpercentile(v, 99.99)
        if peak < 0.5:            # حماية إضافية: قيم شديدة الصغر ⇒ على الأرجح متر
            v = v * 1000.0
            res["meta"]["units_converted"] = "auto-scaled x1000 (assumed metres)"
    src["values"] = v
    return res


def fetch_model_precipitation_24h(model, offline=False):
    """
    إرجاع حقل الأمطار التراكمية 24h (ملم) مُعاد استيفاؤه إلى شبكة التصدير الموحدة.
    المسار: GRIB أصلي → Open-Meteo احتياطي → نمط تخليقي (Offline).
    """
    if offline:
        res = fetch_synthetic_fallback(model)
    else:
        try:
            res = FETCHERS[model]()
        except Exception as e:
            log(f"   ✗ فشل جلب {model} الأصلي: {type(e).__name__}: {shorten(e)}")
            try:
                res = fetch_openmeteo_fallback(model)
            except Exception as e2:
                log(f"   ✗ فشل الاحتياطي أيضاً ({type(e2).__name__}: {shorten(e2)}) — تحويل إلى نمط تخليقي")
                res = fetch_synthetic_fallback(model)
    res = to_mm(res)
    field = regrid(res["src"]).astype(np.float64)
    field = np.clip(np.nan_to_num(field, nan=0.0), 0.0, None)
    return dict(field=field, meta=res["meta"])


# ==============================================================================
# 5. حساب التجميع الفائق المرجّح (Weighted Super-Ensemble)
# ==============================================================================
def compute_multimodel_ensemble(offline=False):
    log("🌧  بدء جلب حقول النماذج الأربعة وحساب التجميع الفائق (MME)...")
    individual_fields = {}
    metas = {}
    ensemble_total_24h = np.zeros_like(GLAT, dtype=np.float64)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut = {m: ex.submit(fetch_model_precipitation_24h, m, offline) for m in WEIGHTS}
        results = {m: f.result() for m, f in fut.items()}

    for model, weight in WEIGHTS.items():
        field = results[model]
        individual_fields[model] = field["field"]
        metas[model] = field["meta"]
        ensemble_total_24h += weight * field["field"]

    # تصفير الضوضاء (الأمطار الأقل من 0.2 ملم تُلغى تلافياً للخداع البصري)
    ensemble_total_24h[ensemble_total_24h < 0.2] = 0.0
    return individual_fields, ensemble_total_24h, metas



# ==============================================================================
# 6ب) حدود مديريات محافظة حضرموت (GADM 4.1 — المستوى الإداري الثاني)
# ==============================================================================
SOCOTRA_DISTRICTS = {"Hidaybu", "QulensyaWaAbdAlKuri"}   # أرخبيل سقطرى محافظة منفصلة منذ 2013
GADM_YEM_ADM2 = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_YEM_2.json.zip"


def load_yemen_districts():
    """
    كل مديريات اليمن (332 مديرية / 21 محافظة) من GADM 4.1 ADM2.
    تُحمَّل مرة واحدة وتُخزَّن في data_cache بصيغة GeoJSON مخففة الخصائص.
    """
    from shapely.geometry import shape
    import json as _json
    cache = cache_path("gadm41_YEM_2_full.geojson")
    if not os.path.exists(cache):
        import zipfile
        raw = http_get(GADM_YEM_ADM2, timeout=300).content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            gj = _json.loads(z.read(z.namelist()[0]))
        feats = [{"type": "Feature",
                  "properties": {"NAME_1": f["properties"].get("NAME_1"),
                                 "NAME_2": f["properties"].get("NAME_2")},
                  "geometry": f["geometry"]} for f in gj["features"]]
        with open(cache, "w") as fh:
            _json.dump({"type": "FeatureCollection", "features": feats}, fh)
        log(f"   🗺 حُدِّث ملف حدود مديريات اليمن من GADM 4.1: {len(feats)} مديرية")
    else:
        feats = _json.load(open(cache))["features"]
    return [(f["properties"]["NAME_1"], f["properties"]["NAME_2"], shape(f["geometry"]))
            for f in feats]


def load_hadramout_districts():
    """مديريات محافظة حضرموت الـ28 (بعد استثناء مديريتي سقطرى)."""
    return [(n, g) for (gov, n, g) in load_yemen_districts()
            if gov == "Hadramawt" and n not in SOCOTRA_DISTRICTS]


# --------------------------- مقاييس الألوان للمنتجين ---------------------------
# ==============================================================================
# 5ب) حداثة البيانات (آخر تحديث لكل نموذج)
# ==============================================================================
def run_datetime(meta):
    """تاريخ دورة التشغيل من نص meta['run'] إن كان أصلياً (صيغة YYYYMMDD/HHZ)."""
    m = re.search(r"(\d{8})/(\d{2})Z", str(meta.get("run", "")))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)


def freshness_lines(metas, metas10=None):
    """سطور تفصيلية: دورة كل نموذج وعمرها بالساعات وقت التوليد."""
    now = utc_now()
    out = [f"Data freshness @ {now.strftime('%Y-%m-%d %H:%M UTC')} :"]
    for m in WEIGHTS:
        mt = metas.get(m, {})
        dt = run_datetime(mt)
        if dt:
            age = (now - dt).total_seconds() / 3600.0
            out.append(f"  {m:<6}: run {dt.strftime('%Y-%m-%d %HZ')}  (age {age:4.1f} h)  "
                       f"window {mt.get('window','?')}")
        else:
            out.append(f"  {m:<6}: {mt.get('run','?')} — أحدث دورة متاحة عبر Open-Meteo  "
                       f"window {mt.get('window','?')}")
    if metas10:
        out.append("  10-Day outlook sources:")
        for m in WEIGHTS:
            mt = metas10.get(m, {})
            out.append(f"    {m:<6}: {mt.get('run','?')}  window {mt.get('window','?')}")
    return out


def freshness_short(metas):
    """سطر مضغوط بأسفل الخريطة (عربي) يوثّق آخر تحديث لكل نموذج."""
    now = utc_now()
    parts = []
    for m in WEIGHTS:
        dt = run_datetime(metas.get(m, {}))
        if dt:
            age = (now - dt).total_seconds() / 3600.0
            parts.append(f"{AR_MODEL[m]} دورة {LTR(dt.strftime('%m-%d %HZ'))} (قبل {LTR(f'{age:.0f}')} س)")
        else:
            parts.append(f"{AR_MODEL[m]} عبر Open-Meteo (أحدث دورة)")
    return ar("تحديث البيانات: " + " | ".join(parts)
              + f" | أُنشئت {LTR(now.strftime('%Y-%m-%d %H:%M UTC'))}")


PALETTE_WB = [
    # باليتة تشغيلية متعددة الألوان مستوحاة من مفاتيح CFS/WeatherBELL:
    # رمادي للخفيف، أخضر ثم أزرق للمتوسط، أصفر/برتقالي/أحمر للغزير، بني فبنفسجي للمتطرف
    '#c1bfc0', '#838383', '#c1fdb7', '#93fb88', '#4def4b', '#08a206', '#0d64ce', '#50a6fd',
    '#acf2f2', '#fef8a0', '#ffc338', '#fc6701', '#ff380f', '#c30302', '#8d0000', '#5f342d',
    '#b88d86', '#f4ded3', '#c8bfde', '#9c8cbd', '#675199', '#7a007e', '#b102b5', '#d304d6']

SCALES = {
    "24h": dict(
        levels=[0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0,
                12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 250.0],
        colors=PALETTE_WB, png="hadramout_mme_precip_24h.png", unit="mm / 24 h",
        title_ar="خلال 24 ساعة القادمة", unit_ar="24 ساعة",
        cblabel=("Ensemble 24-Hour Total Rainfall (mm) — 24-class hydrological scale "
                 "[GFS 40% | ECMWF 30% | GEM 20% | ICON 10%]")),
    "10d": dict(
        levels=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.0,
                12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 250.0, 400.0],
        colors=PALETTE_WB, png="hadramout_mme_precip_10d.png", unit="mm / 10 d",
        title_ar="إجمالي 10 أيام قادمة", unit_ar="10 أيام",
        cblabel=("Ensemble 10-Day Total Rainfall (mm) — 24-class hydrological scale "
                 "[GFS 40% | ECMWF 30% | GEM 20% | ICON 10%]")),
}


# ==============================================================================
# 5ج) إجمالي الأمطار للعشرة الأيام القادمة (Day 0 → Day 9)
# ==============================================================================
def openmeteo_daily_total(model_key, ndays=10, step=0.5):
    """
    مجموع الهطول اليومي (precipitation_sum) لـ ndays يوماً من Open-Meteo لكل نموذج،
    على شبكة نقاط 0.5° تغطي المجال (+هامش)، مع كاش يومي ودفعات محترمة لحد المعدل.
    """
    start = utc_now().strftime("%Y-%m-%d")
    t0 = datetime.strptime(start, "%Y-%m-%d")
    end = (t0 + timedelta(days=ndays - 1)).strftime("%Y-%m-%d")
    cache_key = cache_path(f"om10d_{model_key}_{start}.npz")
    if os.path.exists(cache_key):
        z = np.load(cache_key)
        log(f"   {model_key} (10d) من الذاكرة المؤقتة ({start}→{end}) "
            f"أقصى قيمة={z['values'].max():.1f} ملم")
        return dict(src=dict(kind="regular", lats=z["lats"], lons=z["lons"], values=z["values"]),
                    meta=dict(shortName="tp", stepRange=f"0-{ndays*24}h", units="mm",
                              run=f"Open-Meteo/{z['model']}", window=f"{start} → {end}"))

    la = np.round(np.arange(LAT_MIN - 1, LAT_MAX + 1 + 1e-9, step), 2)
    lo = np.round(np.arange(LON_MIN - 1, LON_MAX + 1 + 1e-9, step), 2)
    pts = [(a, b) for a in la for b in lo]

    names = OM_MODELS.get(model_key, ["best_match"])
    results, used_model, err = None, None, None
    for name in names:
        out, ok = [], True
        for i in range(0, len(pts), 300):
            ch = pts[i:i + 300]
            params = {
                "latitude": ",".join(f"{a:.2f}" for a, _ in ch),
                "longitude": ",".join(f"{b:.2f}" for _, b in ch),
                "daily": "precipitation_sum", "models": name,
                "start_date": start, "end_date": end, "timezone": "GMT",
            }
            waited, res = 0, None
            while True:
                try:
                    with OM_LOCK:
                        gap = time.time() - OM_LAST_CALL[0]
                        if gap < OM_MIN_GAP:
                            time.sleep(OM_MIN_GAP - gap)
                        OM_LAST_CALL[0] = time.time()
                    r = requests.get(OM_URL, params=params, headers=UA, timeout=180)
                except Exception as e:
                    err = f"{type(e).__name__}"; time.sleep(3); waited += 3
                    if waited > 200: ok = False; break
                    continue
                if r.status_code == 200:
                    res = r.json(); break
                if r.status_code in (414, 429, 500, 502, 503):
                    pause = 62 if r.status_code == 429 else 4
                    if waited + pause > 200:
                        err = f"HTTP {r.status_code}"; ok = False; break
                    time.sleep(pause); waited += pause
                    continue
                err = f"HTTP {r.status_code} {r.text[:90]}"; ok = False; break
            if not ok: break
            out.extend(res if isinstance(res, list) else [res])
            time.sleep(0.4)
        if ok and out:
            results, used_model = out, name
            break
    if not results:
        raise RuntimeError(f"Open-Meteo 10d ({model_key}) فشل: {err}")

    acc = np.zeros((len(la), len(lo)), dtype=np.float64)
    for k, item in enumerate(results):
        i, j = divmod(k, len(lo))
        vals = np.nan_to_num(np.asarray(item["daily"]["precipitation_sum"], dtype=np.float64), nan=0.0)
        acc[i, j] = vals[:ndays].sum()
    np.savez_compressed(cache_key, values=acc[::-1, :], lats=la[::-1], lons=lo, model=used_model)
    log(f"   {model_key} (10d) via Open-Meteo[{used_model}]  نقاط={len(pts)}  "
        f"{start}→{end}  أقصى قيمة={acc.max():.1f} ملم")
    return dict(src=dict(kind="regular", lats=la[::-1], lons=lo, values=acc[::-1, :]),
                meta=dict(shortName="tp", stepRange=f"0-{ndays*24}h", units="mm",
                          run=f"Open-Meteo/{used_model}", window=f"{start} → {end}"))


def compute_10d_ensemble(offline=False):
    log("📅 بدء حساب إجمالي أمطار العشرة الأيام القادمة (MME-10d)...")
    from concurrent.futures import ThreadPoolExecutor

    def fetch_one(model):
        if offline:
            rng = np.random.default_rng(abs(hash(model)) % 2**31)
            base = fetch_synthetic_fallback(model)["src"]["values"]
            return dict(field=base * rng.uniform(3, 7), meta=dict(
                shortName="tp", stepRange="0-240h", units="mm",
                run=f"SYNTHETIC/{model}", window="n/a"))
        try:
            res = openmeteo_daily_total(model)
        except Exception as e:
            log(f"   ✗ فشل {model} (10d): {type(e).__name__}: {shorten(e)} — نمط تخليقي")
            rng = np.random.default_rng(abs(hash(model)) % 2**31)
            return dict(field=fetch_synthetic_fallback(model)["src"]["values"] * rng.uniform(3, 7),
                        meta=dict(shortName="tp", stepRange="0-240h", units="mm",
                                  run=f"SYNTHETIC/{model}", window="n/a"))
        field = regrid(res["src"]).astype(np.float64)
        field = np.clip(np.nan_to_num(field, nan=0.0), 0.0, None)
        return dict(field=field, meta=res["meta"])

    with ThreadPoolExecutor(max_workers=4) as ex:
        fut = {m: ex.submit(fetch_one, m) for m in WEIGHTS}
        results = {m: f.result() for m, f in fut.items()}

    individual, metas = {}, {}
    total = np.zeros_like(GLAT, dtype=np.float64)
    for m, w in WEIGHTS.items():
        individual[m] = results[m]["field"]
        metas[m] = results[m]["meta"]
        total += w * results[m]["field"]
    total[total < 0.2] = 0.0
    return individual, total, metas


def darken_hex(hex_color, factor=0.6):
    """تعتيم لون HEX بنسبة factor مع الحفاظ على صبغته (لقراءة الأرقام الملونة)."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


def draw_wbell_key(fig, sc, maxval):
    """مفتاح ألوان بأسلوب النشرات التشغيلية (CFS/WeatherBELL كالصورة المرجعية):
    شريط أفقي عريض أسفل الخريطة بكامل عرضها، كتل فئات منفصلة، الأرقام فوق حدود
    الفئات، وقيمة الأقصى بالخط العريض يميناً."""
    import matplotlib.patches as mpatches
    levels, colors = sc["levels"], sc["colors"]
    nb = len(levels) - 1
    x0, x1 = 0.020, 0.865
    y0, h = 0.016, 0.030
    w = (x1 - x0) / (nb + 1)
    # كتلة "لا أمطار" بيضاء ثم كتل الفئات
    fig.patches.append(mpatches.Rectangle(
        (x0, y0), w, h, transform=fig.transFigure,
        facecolor='#ffffff', edgecolor='#b5b5b5', linewidth=0.5, zorder=8))
    for i in range(nb):
        fig.patches.append(mpatches.Rectangle(
            (x0 + w * (i + 1), y0), w, h, transform=fig.transFigure,
            facecolor=colors[i], edgecolor='none', zorder=8))
    # الأرقام فوق حدود الفئات (بداية كل كتلة + الحد الأيمن للأخيرة)
    for j in range(nb + 1):
        fig.text(x0 + w * (j + 1), y0 + h + 0.005, f"{levels[j]:g}",
                 fontsize=6.5, color='#111111', ha='center', va='bottom', zorder=9)
    # الأقصى يميناً بخط عريض (كما في المرجع)
    fig.text(0.988, y0 + h / 2, ar(f"الأقصى: {LTR(f'{maxval:.2f}')} ملم"),
             fontsize=9.5, fontweight='bold', color='#000000',
             ha='right', va='center', zorder=9)



def plot_multi_model_map(ensemble_field, metas=None, outfile=None, product="24h"):
    sc = SCALES[product]
    outfile = outfile or os.path.join(OUT_DIR, sc["png"])
    print(f"🎨 توليد لوحة التوقعات المدمجة المجمعة ({product} Accumulated Precipitation)...")
    fig = plt.figure(figsize=(14, 10), dpi=300)
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)

    # معالم الخريطة
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#faf8f5', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#f5f8ff', zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#f5f8ff',
                   edgecolor='#9bb7c9', linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.2, edgecolor='#1a1a1a', zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='--', edgecolor='#555555', zorder=5)

    # حدود المديريات: حضرموت بخط بنّي واضح، وبقية اليمن بخطوط أرفع،
    # إضافة إلى إطارات المحافظات (اتحاد مديريات كل محافظة)
    try:
        from shapely.ops import unary_union
        all_dist = load_yemen_districts()
        had_geoms = [g for (gov, n, g) in all_dist
                     if gov == "Hadramawt" and n not in SOCOTRA_DISTRICTS]
        other_geoms = [g for (gov, n, g) in all_dist if gov != "Hadramawt"]
        ax.add_feature(cfeature.ShapelyFeature(
            other_geoms, proj, facecolor='none', edgecolor='#a1887f',
            linewidth=0.55, linestyle=(0, (3, 2)), zorder=6))
        gov_geoms = {}
        for (gov, n, g) in all_dist:
            gov_geoms.setdefault(gov, []).append(g)
        unions = [unary_union(gs) for gs in gov_geoms.values()]
        ax.add_feature(cfeature.ShapelyFeature(
            unions, proj, facecolor='none', edgecolor='#6d5a4a',
            linewidth=1.1, zorder=6.5))
        ax.add_feature(cfeature.ShapelyFeature(
            had_geoms, proj, facecolor='none', edgecolor='#7b4a21',
            linewidth=1.05, linestyle=(0, (5, 2)), zorder=6.6))
        ax.add_feature(cfeature.ShapelyFeature(
            [unary_union(had_geoms)], proj, facecolor='none', edgecolor='#5d4037',
            linewidth=1.6, zorder=7))
        log(f"   🗺 رُسمت المديريات: {len(had_geoms)} لحضرموت + {len(other_geoms)} لبقية اليمن "
            f"({len(unions)} محافظة)")
    except Exception as e:
        log(f"   ⚠ تعذّر رسم حدود المديريات: {type(e).__name__}: {shorten(e)}")

    levels, colors = sc["levels"], sc["colors"]
    cmap = ListedColormap(colors)

    cf = ax.contourf(
        GLON, GLAT, ensemble_field,
        levels=levels,
        colors=colors,         # لون مستقل ومطابق تماماً لكل فئة (16 فئة + فيضان)
        extend='max',
        alpha=0.88,
        zorder=3,
        transform=proj
    )
    # حدود الفئات لوضوح القراءة الكارتوجرافية
    ax.contour(GLON, GLAT, ensemble_field, levels=levels, colors='#404040',
               linewidths=0.4, alpha=0.6, zorder=4, transform=proj)

    # مفتاح الخريطة بأسلوب meteologix.com
    draw_wbell_key(fig, sc, float(np.nanmax(ensemble_field)))
    if metas:
        fig.text(0.99, 0.066, freshness_short(metas), fontsize=7,
                 color='#333333', ha='right', zorder=9)
    # سطرا معلومات أعلى اليسار (مقابل العنوان العربي يميناً)
    fig.text(0.012, 0.992, ar("تصميم: أحمد عمر ظافر"),
             fontsize=8.5, fontweight='bold', color='#333333', ha='left', va='top')
    fig.text(0.012, 0.970,
             ar("الدمج متعدد النماذج — " + " · ".join(
                 f"{AR_MODEL[m]} {LTR(f'{WEIGHTS[m]*100:.0f}%')}" for m in WEIGHTS)),
             fontsize=7.5, color='#444444', ha='left', va='top')

    # (أُزيلت نقاط المحطات ولافتاتها من الخريطة بناءً على الطلب —
    #  قيم المحطات تبقى متوفرة في ملف CSV وسجل التقرير)

    # شبكة الإحداثيات
    gl = ax.gridlines(draw_labels=True, linestyle=':', alpha=0.5, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    # عنوان مختصر بالعربية (سطران)
    if product == "10d":
        # العنوان المطلوب: إجمالي الأمطار التراكمية 10 أيام قادمة من تاريخ إلى تاريخ
        d0 = reference_day()
        d1 = (datetime.strptime(d0, "%Y-%m-%d") + timedelta(days=9)).strftime("%Y-%m-%d")
        line1 = f"إجمالي الأمطار التراكمية 10 أيام قادمة من {LTR(d0)} إلى {LTR(d1)}"
        line2 = "حضرموت وخليج عدن  |  دمج متعدد النماذج (MME)"
    else:
        win_txt = f"{reference_day()} 00Z → +24h"
        line1 = "توقعات الأمطار التراكمية — حضرموت وخليج عدن"
        line2 = f"{sc['title_ar']}  |  {LTR(win_txt)}  |  دمج متعدد النماذج (MME)"
    ax.set_title(ar(line1) + "\n" + ar(line2),
                 fontsize=12, fontweight='bold', loc='right', pad=10)
    plt.tight_layout(rect=[0, 0.085, 1, 1])
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    print(f"✅ تم تصدير خريطة التنبؤ التجميعي بنجاح: {outfile}")
    return outfile


# ==============================================================================
# 7. تقارير مساعدة (CSV + سجل تشغيل)
# ==============================================================================
STATIONS = {
    'Al-Mukalla': (49.12, 14.53),
    'Al-Dhabba Port': (49.50, 14.70),
    'Ash-Shihr': (49.61, 14.76),
    'Sayun': (48.78, 15.93),
    'Tarim': (49.00, 16.05),
    'Wadi Doan': (48.35, 15.05),
    'Wadi Huwayrah': (49.42, 14.88),
}

# إزاحة لافتة كل محطة (dx, dy) بالدرجات — تُعاير يدوياً لمنع تراكب اللافتات
STATION_LABEL_OFFSETS = {
    'Al-Mukalla': (-1.00, -0.75),
    'Al-Dhabba Port': (-1.05, -0.15),
    'Ash-Shihr': (0.15, -0.35),
    'Sayun': (-1.10, 0.00),
    'Tarim': (0.15, 0.10),
    'Wadi Doan': (-1.05, -0.05),
    'Wadi Huwayrah': (0.15, 0.15),
}


def value_at(field, lon_st, lat_st):
    return field[np.argmin(np.abs(grid_lats - lat_st)), np.argmin(np.abs(grid_lons - lon_st))]


def export_station_csv(individual, ensemble, metas, fname="hadramout_mme_stations_24h.csv",
                       tag="MME_24h_mm"):
    out = os.path.join(OUT_DIR, fname)
    header = ["station", "lon", "lat"] + list(WEIGHTS) + [tag]
    with open(out, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for name, (lon_st, lat_st) in STATIONS.items():
            row = [name.replace(" ", "_"), f"{lon_st:.2f}", f"{lat_st:.2f}"]
            row += [f"{value_at(individual[m], lon_st, lat_st):.2f}" for m in WEIGHTS]
            row += [f"{value_at(ensemble, lon_st, lat_st):.2f}"]
            f.write(",".join(row) + "\n")
    log(f"📄 تم تصدير قيم المحطات: {out}")
    return out


def write_report(individual, ensemble, metas, png, csvf, elapsed, extra=None):
    out = os.path.join(OUT_DIR, "mme_report.txt")
    lines = []
    lines.append("=" * 78)
    lines.append(" Multi-Model Super-Ensemble (MME) — Hadramout / Gulf of Aden — 24h Rainfall")
    lines.append("=" * 78)
    lines.append(f"Generated (UTC) : {utc_now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Domain          : lon [{LON_MIN}, {LON_MAX}]  lat [{LAT_MIN}, {LAT_MAX}]")
    lines.append(f"Target grid     : {TARGET_RES}° -> {GLAT.shape[0]} x {GLAT.shape[1]} "
                 f"({GLAT.size} node)")
    lines.append(f"Run time        : {elapsed:.1f} s")
    lines.append(f"Noise threshold : values < 0.2 mm set to 0")
    lines.append("")
    lines.append("-- Model provenance --------------------------------------------------------")
    for m in WEIGHTS:
        mt = metas.get(m, {})
        f_ = individual[m]
        lines.append(f"  {m:<6} w={WEIGHTS[m]:.2f}  run={mt.get('run','?'):<16} "
                     f"param={mt.get('shortName','?'):<4} step={mt.get('stepRange','?'):<6} "
                     f"units={mt.get('units','?'):<7} window={mt.get('window','?'):<18} "
                     f"max={f_.max():7.2f} mm  mean={f_.mean():5.2f} mm  "
                     f"wet%={100*(f_ >= 0.2).mean():5.1f}")
    lines.append("")
    lines.append("-- Ensemble statistics -----------------------------------------------------")
    lines.append(f"  max  = {ensemble.max():.2f} mm   at "
                 f"lat {grid_lats[np.unravel_index(ensemble.argmax(), ensemble.shape)[0]]:.2f}, "
                 f"lon {grid_lons[np.unravel_index(ensemble.argmax(), ensemble.shape)[1]]:.2f}")
    lines.append(f"  mean = {ensemble.mean():.2f} mm  (over full domain)")
    wet = ensemble >= 0.2
    lines.append(f"  wet fraction = {100*wet.mean():.1f} % of domain")
    lines.append(f"  areal mean over wet cells = {ensemble[wet].mean() if wet.any() else 0:.2f} mm")
    lines.append("")
    lines.append("-- Station values (mm / 24h) ----------------------------------------------")
    lines.append(f"  {'Station':<16}" + "".join(f"{m:>8}" for m in WEIGHTS) + f"{'MME':>9}")
    for name, (lon_st, lat_st) in STATIONS.items():
        lines.append(f"  {name:<16}"
                     + "".join(f"{value_at(individual[m], lon_st, lat_st):8.2f}" for m in WEIGHTS)
                     + f"{value_at(ensemble, lon_st, lat_st):9.2f}")
    lines.append("")
    if extra:
        for title, body in extra:
            lines.append(f"-- {title} " + "-" * max(2, 74 - len(title)))
            lines.extend(body)
            lines.append("")
    lines.append(f"-- Outputs -----------------------------------------------------------------")
    lines.append(f"  map : {png}")
    lines.append(f"  csv : {csvf}")
    lines.append("=" * 78)
    txt = "\n".join(lines)
    open(out, "w", encoding="utf-8").write(txt + "\n")
    print("\n" + txt + "\n")
    log(f"🧾 تم حفظ سجل التشغيل: {out}")
    return out


# ==============================================================================
# 8. التنفيذ المباشر
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Hadramout Multi-Model Ensemble rainfall (24h + 10d)")
    ap.add_argument("--offline", action="store_true",
                    help="تشغيل بدون إنترنت باستخدام حقول تخليقية (لاختبار المسار فقط)")
    ap.add_argument("--outfile", default=None, help="مسار ملف خريطة الـ24h الناتجة")
    args = ap.parse_args()

    if eccodes is None and not args.offline:
        log("⚠ مكتبة eccodes غير متوفرة — سيُستخدم Open-Meteo/النمط التخليقي")

    t0 = time.time()
    # ---- منتج 24 ساعة ----
    fields, ensemble_precip, metas = compute_multimodel_ensemble(offline=args.offline)
    png = plot_multi_model_map(ensemble_precip, metas=metas, outfile=args.outfile, product="24h")
    csvf = export_station_csv(fields, ensemble_precip, metas)

    # ---- منتج 10 أيام ----
    f10, ens10, metas10 = compute_10d_ensemble(offline=args.offline)
    png10 = plot_multi_model_map(ens10, metas=metas10, product="10d")
    csv10 = export_station_csv(f10, ens10, metas10,
                               fname="hadramout_mme_stations_10d.csv", tag="MME_10d_mm")

    # ---- حداثة البيانات ----
    fresh = freshness_lines(metas, metas10)
    print("\n" + "\n".join(fresh) + "\n")

    stats10 = [
        f"  window   = {metas10['ECMWF'].get('window','?')} (10 UTC days)",
        f"  max      = {ens10.max():.2f} mm at lat "
        f"{grid_lats[np.unravel_index(ens10.argmax(), ens10.shape)[0]]:.2f}, "
        f"lon {grid_lons[np.unravel_index(ens10.argmax(), ens10.shape)[1]]:.2f}",
        f"  mean     = {ens10.mean():.2f} mm over domain | wet% = {100*(ens10 >= 0.2).mean():.1f}",
        f"  map      = {png10}",
        f"  csv      = {csv10}",
    ]
    write_report(fields, ensemble_precip, metas, png, csvf, time.time() - t0,
                 extra=[("10-Day ensemble (Day 0 → Day 9)", stats10),
                        ("Data freshness (آخر تحديث البيانات)", fresh)])
    return png


if __name__ == "__main__":
    main()
