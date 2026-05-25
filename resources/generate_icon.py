"""
Генерация иконки приложения (CT Reconstruction Viewer).

Создаёт icon.png (256x256) и icon.ico (multi-resolution 16/32/48/64/128/256)
рядом со скриптом, рисуя иконку программно через PIL — БЕЗ cairo/svg
(чтобы не было проблем с системными DLL на Windows).

Зависимости:
    pip install pillow

Запуск:
    python resources/generate_icon.py
"""
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Нужен Pillow: pip install pillow")


HERE = Path(__file__).parent
PNG_PATH = HERE / "icon.png"
ICO_PATH = HERE / "icon.ico"

SIZE = 256

# Цвета (RGBA)
BG_DARK         = (30, 30, 46, 255)     # фон
ACCENT          = (255, 183, 77, 255)   # оранжевые лучи + центр
ACCENT_DIM      = (255, 183, 77, 140)
SKULL_OUTLINE   = (224, 224, 224, 255)  # ободок "черепа"
SKULL_FILL      = (42, 42, 62, 255)     # внутренний цвет черепа
BRAIN_FILL      = (168, 179, 199, 255)  # ткань мозга
VENTRICLES      = (30, 30, 46, 255)     # тёмные внутренние области


def make_icon(size: int = SIZE) -> Image.Image:
    """Рисует иконку CT-среза с радиальными лучами рентгена."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Скруглённый фон
    corner = int(size * 0.16)
    d.rounded_rectangle([0, 0, size - 1, size - 1], corner, fill=BG_DARK)

    cx, cy = size / 2, size / 2

    # Радиальные лучи рентгена — 8 штук
    n_rays = 8
    inner_r = size * 0.04
    outer_r = size * 0.47
    ray_w = max(1, size // 128)
    for i in range(n_rays):
        ang = 2 * math.pi * i / n_rays
        x1 = cx + inner_r * math.cos(ang)
        y1 = cy + inner_r * math.sin(ang)
        x2 = cx + outer_r * math.cos(ang)
        y2 = cy + outer_r * math.sin(ang)
        d.line([(x1, y1), (x2, y2)], fill=ACCENT_DIM, width=ray_w)

    # Тонкое кольцо gantry
    gantry_r = size * 0.415
    d.ellipse(
        [cx - gantry_r, cy - gantry_r, cx + gantry_r, cy + gantry_r],
        outline=ACCENT, width=max(2, size // 90),
    )

    # Череп (внешнее кольцо)
    skull_r = size * 0.32
    d.ellipse(
        [cx - skull_r, cy - skull_r, cx + skull_r, cy + skull_r],
        outline=SKULL_OUTLINE, fill=SKULL_FILL, width=max(2, size // 64),
    )

    # Серое вещество (внутренний эллипс)
    brain_rx = size * 0.24
    brain_ry = size * 0.225
    d.ellipse(
        [cx - brain_rx, cy - brain_ry, cx + brain_rx, cy + brain_ry],
        fill=BRAIN_FILL,
    )

    # Желудочки (тёмные) — упрощённо два эллипса
    v_off_x = size * 0.07
    v_off_y = size * 0.04
    v_rx = size * 0.035
    v_ry = size * 0.08
    d.ellipse(
        [cx - v_off_x - v_rx, cy - v_off_y - v_ry,
         cx - v_off_x + v_rx, cy - v_off_y + v_ry],
        fill=VENTRICLES,
    )
    d.ellipse(
        [cx + v_off_x - v_rx, cy - v_off_y - v_ry,
         cx + v_off_x + v_rx, cy - v_off_y + v_ry],
        fill=VENTRICLES,
    )
    # Третий "желудочек" по центру
    d.ellipse(
        [cx - size * 0.024, cy + size * 0.04 - size * 0.04,
         cx + size * 0.024, cy + size * 0.04 + size * 0.04],
        fill=VENTRICLES,
    )

    # Центральная точка
    dot_r = max(2, size // 80)
    d.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=ACCENT,
    )

    return img


def main():
    print("Рисую иконку 256x256...")
    img = make_icon(SIZE)
    img.save(PNG_PATH, format="PNG")
    print(f"  PNG: {PNG_PATH} ({PNG_PATH.stat().st_size} bytes)")

    # Multi-resolution ICO — Pillow сам пересэмплирует под нужные размеры.
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ICO_PATH, format="ICO", sizes=sizes)
    print(f"  ICO: {ICO_PATH} ({ICO_PATH.stat().st_size} bytes) — sizes: {sizes}")
    print("\nГотово. Не забудьте re-configure CMake:")
    print("  cmake -S . -B out\\build\\release -DCMAKE_BUILD_TYPE=Release")


if __name__ == "__main__":
    main()
