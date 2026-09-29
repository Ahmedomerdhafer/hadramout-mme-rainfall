#!/usr/bin/env python3
"""تحديث صفحة المعاينة outputs/index.html تلقائياً من outputs/mme_report.txt."""
import re, io

rep = io.open("outputs/mme_report.txt", encoding="utf-8").read()
html = io.open("outputs/index.html", encoding="utf-8").read()

ts = re.search(r"Data freshness @ ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} UTC)", rep).group(1)
gfs = re.search(r"GFS\s+: run (\S+ \S+Z)\s+\(age\s+([\d.]+) h\)\s+window (\S+ \S+)", rep)
ecm = re.search(r"ECMWF\s+: run (\S+ \S+Z)\s+\(age\s+([\d.]+) h\)\s+window (\S+ \S+)", rep)
ens = re.search(r"max\s+=\s+([\d.]+) mm\s+at lat ([\d.]+), lon ([\d.]+)\s*\n\s*mean = ([\d.]+) mm[^\n]*\n\s*wet fraction = ([\d.]+)", rep)
d10 = re.search(r"10-Day ensemble.*?max\s+=\s+([\d.]+) mm at lat ([\d.]+), lon ([\d.]+)\s*\n\s*mean\s+=\s+([\d.]+) mm over domain \| wet% = ([\d.]+)", rep, re.S)
w10 = re.search(r"window\s+= ([0-9]{4}-[0-9]{2}-[0-9]{2}) → ([0-9]{4}-[0-9]{2}-[0-9]{2})", rep)
gem = re.search(r"GEM\s+: Open-Meteo/\S+[^\n]*window (\S+ \S+)", rep)
html = re.sub(r"زمن إنشاء المنتج: <b>[^<]*</b>", f"زمن إنشاء المنتج: <b>{ts}</b>", html)
html = re.sub(r"والألماني ICON: أحدث دورة متاحة عبر Open-Meteo — نافذة [^<]+",
              f"والألماني ICON: أحدث دورة متاحة عبر Open-Meteo — نافذة {gem.group(1)}", html)
html = re.sub(r"إجمالي الأمطار التراكمية 10 أيام قادمة \([^)]*\)",
              f"إجمالي الأمطار التراكمية 10 أيام قادمة ({w10.group(1)} → {w10.group(2)})", html)
refday = gem.group(1).split()[0]
html = re.sub(r"خريطة 24 ساعة القادمة \([^)]*\)",
              f"خريطة 24 ساعة القادمة ({refday} 00Z → +24h)", html)

# إحصاء 24 ساعة (أول مجموعة stats) ثم 10 أيام (الثانية)
html = re.sub(r"الأمريكي GFS: دورة <b>[^<]*</b> \(قبل [\d.]+ ساعة\) — نافذة [^<]+",
              f"الأمريكي GFS: دورة <b>{gfs.group(1)}</b> (قبل {float(gfs.group(2)):g} ساعة) — نافذة {gfs.group(3)}", html)
html = re.sub(r"الأوروبي ECMWF: دورة <b>[^<]*</b> \(قبل [\d.]+ ساعة\) — نافذة [^<]+",
              f"الأوروبي ECMWF: دورة <b>{ecm.group(1)}</b> (قبل {float(ecm.group(2)):g} ساعة) — نافذة {ecm.group(3)}", html)
html = re.sub(r"نافذة توقعات 10 أيام \(النماذج الأربعة عبر Open-Meteo\): <b>[^<]*</b>",
              f"نافذة توقعات 10 أيام (النماذج الأربعة عبر Open-Meteo): <b>{w10.group(1)} → {w10.group(2)}</b>", html)

# إحصاء 24 ساعة (أول مجموعة stats) ثم 10 أيام (الثانية)
stats24 = f'''<div class="stat">أقصى مجموع MME<b>{ens.group(1)} ملم</b>عند {ens.group(2)}°ش، {ens.group(3)}°شر</div>
    <div class="stat">المتوسط المجالي<b>{ens.group(4)} ملم</b></div>
    <div class="stat">نسبة المساحة الممطرة<b>{ens.group(5)}%</b></div>'''
stats10 = f'''<div class="stat">أقصى مجموع MME<b>{d10.group(1)} ملم</b>عند {d10.group(2)}°ش، {d10.group(3)}°شر</div>
    <div class="stat">المتوسط المجالي<b>{d10.group(4)} ملم</b></div>
    <div class="stat">نسبة المساحة الممطرة<b>{d10.group(5)}%</b></div>'''
pat = r'<div class="stats">.*?</div>\s*</div>\s*</div>'
# نستبدل المجموعتين بالترتيب
blocks = list(re.finditer(r'<div class="stats">(?:\s*<div class="stat">.*?</div>){3}\s*</div>', html, re.S))
assert len(blocks) == 2, f"stats blocks={len(blocks)}"
html = html[:blocks[0].start()] + '<div class="stats">\n    ' + stats24 + '\n  </div>' + \
       html[blocks[0].end():blocks[1].start()] + '<div class="stats">\n    ' + stats10 + '\n  </div>' + html[blocks[1].end():]

io.open("outputs/index.html", "w", encoding="utf-8").write(html)
print("index updated:", ts, "| GFS", gfs.group(1), "| ECMWF", ecm.group(1),
      "| 24h max", ens.group(1), "| 10d max", d10.group(1))
