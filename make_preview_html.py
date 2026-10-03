#!/usr/bin/env python3
"""بناء ملف معاينة ذاتي الاكتفاء outputs/preview.html:
خرائط مضمّنة base64 + حداثة البيانات + الإحصاءات + جداول المحطات + روابط المعاينة الحية.
يعمل دون أي شبكة (يفتح في المتصفح أو عارض الملفات أو يُشارك كملف واحد)."""
import base64, csv, io, os, re

OUT = "outputs"
rep = io.open(f"{OUT}/mme_report.txt", encoding="utf-8").read()

ts = re.search(r"Data freshness @ ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} UTC)", rep).group(1)
gfs = re.search(r"GFS\s+: run (\S+ \S+Z)\s+\(age\s+([\d.]+) h\)\s+window (\S+ \S+)", rep)
ecm = re.search(r"ECMWF\s+: run (\S+ \S+Z)\s+\(age\s+([\d.]+) h\)\s+window (\S+ \S+)", rep)
gem = re.search(r"GEM\s+: Open-Meteo/\S+[^\n]*window (\S+ \S+)", rep)
ens = re.search(r"max\s+=\s+([\d.]+) mm\s+at lat ([\d.]+), lon ([\d.]+)\s*\n\s*mean = ([\d.]+) mm[^\n]*\n\s*wet fraction = ([\d.]+)", rep)
d10 = re.search(r"10-Day ensemble.*?max\s+=\s+([\d.]+) mm at lat ([\d.]+), lon ([\d.]+)\s*\n\s*mean\s+=\s+([\d.]+) mm over domain \| wet% = ([\d.]+)", rep, re.S)
# يقبل كلا الشكلين في التقرير: "window = ..." و"window ...".
w10 = re.search(r"window\s*(?:=\s*)?([0-9]{4}-[0-9]{2}-[0-9]{2})\s+→\s+([0-9]{4}-[0-9]{2}-[0-9]{2})", rep)

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img24 = b64(f"{OUT}/hadramout_mme_precip_24h.png")
img10 = b64(f"{OUT}/hadramout_mme_precip_10d.png")

def stations(csvpath, mcol):
    rows = []
    with open(csvpath, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["station"].replace("_", " "), r["GFS"], r["ECMWF"], r["GEM"], r["ICON"], r[mcol]))
    return rows

st24 = stations(f"{OUT}/hadramout_mme_stations_24h.csv", "MME_24h_mm")
st10 = stations(f"{OUT}/hadramout_mme_stations_10d.csv", "MME_10d_mm")

def table(rows, unit):
    tr = "".join(
        f"<tr><td>{n}</td><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td class='m'>{m}</td></tr>"
        for n, a, b, c, d, m in rows)
    return (f"<table><tr><th>المحطة</th><th>GFS</th><th>ECMWF</th><th>GEM</th><th>ICON</th>"
            f"<th>MME ({unit})</th></tr>{tr}</table>")

sb = os.environ.get("E2B_SANDBOX_ID", "")
live = f"https://8000-{sb}.e2b.app/" if sb else "#"

html = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>معاينة الأمطار التراكمية — حضرموت وخليج عدن | تصميم: أحمد عمر ظافر</title>
<style>
 body{{font-family:'Segoe UI',Tahoma,Arial,sans-serif;background:#f2f5f8;color:#1c2733;margin:0;padding:20px}}
 header{{background:#0b3d66;color:#fff;border-radius:12px;padding:16px 20px;margin-bottom:16px}}
 header h1{{margin:0 0 6px;font-size:21px}}
 header .sub{{font-size:13px;opacity:.92}}
 .card{{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.12);padding:14px 16px;margin-bottom:16px}}
 .card h2{{margin:0 0 10px;font-size:16.5px;color:#0b3d66}}
 .fresh{{font-size:13.5px;line-height:1.9}}
 .fresh b{{color:#0b3d66}}
 .stats{{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}}
 .stat{{background:#eef4fa;border-radius:8px;padding:8px 14px;font-size:13px}}
 .stat b{{display:block;font-size:16px;color:#0b3d66}}
 img{{width:100%;border-radius:8px;border:1px solid #d5dde5;display:block}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{border:1px solid #dfe6ec;padding:5px 8px;text-align:center}}
 th{{background:#eef4fa;color:#0b3d66}}
 td.m{{font-weight:bold;color:#0b3d66}}
 a{{color:#0b62a8}}
 .note{{font-size:12px;color:#5a6b7c;margin-top:8px}}
 footer{{font-size:12.5px;color:#5a6b7c;text-align:center;margin-top:6px}}
</style>
</head>
<body>
<header>
 <h1>إجمالي الأمطار التراكمية — حضرموت وخليج عدن</h1>
 <div class="sub">دمج متعدد النماذج (MME): الأمريكي GFS 40% · الأوروبي ECMWF 30% · الكندي GEM 20% · الألماني ICON 10% &nbsp;|&nbsp; تصميم: أحمد عمر ظافر</div>
</header>

<div class="card fresh">
 <h2>آخر تحديث للبيانات</h2>
 زمن إنشاء المنتج: <b>{ts}</b><br>
 الأمريكي GFS: دورة <b>{gfs.group(1)}</b> (قبل {float(gfs.group(2)):g} ساعة) — نافذة {gfs.group(3)}<br>
 الأوروبي ECMWF: دورة <b>{ecm.group(1)}</b> (قبل {float(ecm.group(2)):g} ساعة) — نافذة {ecm.group(3)}<br>
 الكندي GEM والألماني ICON: أحدث دورة عبر Open-Meteo — نافذة {gem.group(1)}<br>
 نافذة توقعات 10 أيام: <b>{w10.group(1)} → {w10.group(2)}</b>
</div>

<div class="card">
 <h2>خريطة 24 ساعة القادمة</h2>
 <div class="stats">
  <div class="stat">أقصى مجموع MME<b>{ens.group(1)} ملم</b>عند {ens.group(2)}°ش، {ens.group(3)}°شر</div>
  <div class="stat">المتوسط المجالي<b>{ens.group(4)} ملم</b></div>
  <div class="stat">نسبة المساحة الممطرة<b>{ens.group(5)}%</b></div>
 </div>
 <img src="data:image/png;base64,{img24}" alt="خريطة 24 ساعة">
 <h2 style="margin-top:14px">قيم المحطات (ملم / 24 ساعة)</h2>
 {table(st24, "24h")}
</div>

<div class="card">
 <h2>إجمالي الأمطار التراكمية 10 أيام قادمة ({w10.group(1)} → {w10.group(2)})</h2>
 <div class="stats">
  <div class="stat">أقصى مجموع MME<b>{d10.group(1)} ملم</b>عند {d10.group(2)}°ش، {d10.group(3)}°شر</div>
  <div class="stat">المتوسط المجالي<b>{d10.group(4)} ملم</b></div>
  <div class="stat">نسبة المساحة الممطرة<b>{d10.group(5)}%</b></div>
 </div>
 <img src="data:image/png;base64,{img10}" alt="خريطة 10 أيام">
 <h2 style="margin-top:14px">قيم المحطات (ملم / 10 أيام)</h2>
 {table(st10, "10d")}
</div>

<div class="card">
 <h2>رابط المعاينة الحية (يتجدد مع كل جلسة)</h2>
 <a href="{live}">{live}</a>
 <div class="note">هذا الملف نفسه يعمل دون إنترنت: الصور والبيانات مضمّنة داخله بالكامل، ويمكن مشاركته أو حفظه كما هو.</div>
</div>

<footer>المصادر: ECMWF Open Data · NOAA GFS · DWD ICON · MSC GEM عبر Open-Meteo | شبكة 0.10° | للأغراض البحثية والإرشاد الهيدرولوجي فقط</footer>
</body>
</html>
"""
io.open(f"{OUT}/preview.html", "w", encoding="utf-8").write(html)
print("preview.html written:", os.path.getsize(f"{OUT}/preview.html") // 1024, "KB")
