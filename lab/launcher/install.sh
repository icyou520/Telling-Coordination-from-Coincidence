#!/bin/bash
# Build "Emergence Lab.app" (branded icon) on the Desktop. Double-clicking it rebuilds the
# visual lab from the latest results and opens it in the default browser.
#
#   bash lab/launcher/install.sh            # or: bash lab/launcher/install.sh /path/to/dir
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-$HOME/Desktop}"
APP="$DEST/Emergence Lab.app"
UV="$(command -v uv)"
WORK="$(mktemp -d)"

export MPLCONFIGDIR="$WORK/mpl"
(cd "$PROJECT" && "$UV" run python -m lab.launcher.make_icon "$WORK/icon.png")
mkdir -p "$WORK/AppIcon.iconset"
for s in 16 32 128 256 512; do
  sips -z $s $s "$WORK/icon.png" --out "$WORK/AppIcon.iconset/icon_${s}x${s}.png" >/dev/null
  sips -z $((2 * s)) $((2 * s)) "$WORK/icon.png" --out "$WORK/AppIcon.iconset/icon_${s}x${s}@2x.png" >/dev/null
done

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
iconutil -c icns "$WORK/AppIcon.iconset" -o "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Emergence Lab</string>
  <key>CFBundleDisplayName</key><string>Emergence Lab</string>
  <key>CFBundleIdentifier</key><string>local.lifecomplexity.emergencelab</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>emergence-lab</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSUIElement</key><true/>
  <key>LSArchitecturePriority</key><array><string>$(uname -m)</string></array>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/emergence-lab" <<LAUNCH
#!/bin/bash
cd "$PROJECT"
export MPLCONFIGDIR="\${TMPDIR:-/tmp}/emergence-lab-mpl"
/usr/bin/arch -$(uname -m) "$UV" run python -m lab.build > "\${TMPDIR:-/tmp}/emergence-lab.log" 2>&1
open "$PROJECT/lab/site/index.html"
LAUNCH
chmod +x "$APP/Contents/MacOS/emergence-lab"

touch "$APP"
rm -rf "$WORK"
echo "Installed: $APP"
