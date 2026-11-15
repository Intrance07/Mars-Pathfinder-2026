import json, numpy as np, rasterio
from PIL import Image

DEM = "backend/data/raw/M20_JezeroCrater_CTXDEM_20m (1).tif"
with rasterio.open(DEM) as src:
    z = src.read(1).astype("float64")
    t, nd = src.transform, src.nodata

bad = ~np.isfinite(z)
if nd is not None:
    bad |= (z == nd)
z[bad] = z[~bad].min()

gy, gx = np.gradient(z, abs(t.e), t.a)
az, alt = np.radians(315), np.radians(45)
L = (np.cos(alt)*np.sin(az), np.cos(alt)*np.cos(az), np.sin(alt))
hs = np.clip((-gx*L[0] + gy*L[1] + L[2]) / np.sqrt(gx**2 + gy**2 + 1), 0, 1)

e = (z - z.min()) / (z.max() - z.min())
rgb = np.stack([0.45+0.55*e, 0.20+0.30*e, 0.12+0.10*e], -1) * (0.25+0.75*hs)[..., None]
Image.fromarray((rgb*255).astype("uint8")).save("frontend/public/jezero_basemap.png")

h, w = z.shape
json.dump({"width": w, "height": h, "x0": t.c, "y0": t.f, "pixel": t.a,
           "radius": 3396190, "lat0": 18.4663},
          open("frontend/public/dem_info.json", "w"), indent=2)
print(w, h, t.c, t.f)