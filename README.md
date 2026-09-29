

## النشر والتحديث التلقائي على GitHub

يحتوي هذا المستودع على سير عمل GitHub Actions في
`.github/workflows/update-and-deploy.yml` يقوم بما يلي:

1. تشغيل يومي عند **06:15 UTC**، مع إمكانية التشغيل اليدوي من تبويب Actions.
2. تثبيت الاعتماديات وجلب أحدث دورات GFS وECMWF وICON وGEM.
3. إعادة إنشاء خرائط 24 ساعة و10 أيام، وملفات CSV والتقرير.
4. تحديث `outputs/index.html` و`outputs/preview.html`.
5. نشر مجلد `outputs/` تلقائيًا على GitHub Pages.

يمكن فتح صفحة النتائج من رابط **About → Website** في المستودع. قد يستغرق أول نشر بضع دقائق بعد إنشاء المستودع.
