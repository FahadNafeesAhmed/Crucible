import json
import base64
import re

with open('Crucible Eval Console (standalone).html', 'r', encoding='utf-8') as f:
    content = f.read()

m = re.search(r'<script type="__bundler/manifest">(.*?)</script>', content, re.DOTALL)
manifest = json.loads(m.group(1))
mimes = set(v.get('mime') for v in manifest.values())
print("Mime types:", mimes)

# Also let's check `__bundler/template`
m2 = re.search(r'<script type="__bundler/template">(.*?)</script>', content, re.DOTALL)
if m2:
    print("Found template")
    with open('unpacked_template.json', 'w', encoding='utf-8') as f:
        f.write(m2.group(1))
