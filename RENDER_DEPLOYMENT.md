# Mars Pathfinder deployment on Render

This guide prepares the existing frontend and FastAPI backend as two Render services. The static frontend calls the public API service using a build-time URL. The API image contains the DEM and science-site GeoJSON needed by the terrain engine, so it does not depend on a path on a visitor's computer or on a runtime download.

## Render Free limitations

- The API is configured for Render's Free web service. Free services can spin down after 15 minutes without inbound traffic and take about a minute to start again. Free service hours are limited per workspace, and Render can restart a Free service.
- Free web services have 512 MB RAM and 0.1 CPU, do not support persistent disks, and cannot receive private-network traffic. The landing-point evaluator has been changed to compute the same requested neighborhood from a windowed DEM read; the real-data test measured about 80.5 MiB peak process memory in the local Conda environment. A combined local API-handler run for routes and landing assessment peaked at about 141.1 MiB. Cloud memory and timing can still differ.
- The DEM is part of the backend container image. This makes it available in that deployed image after a process restart or spin-down. It is not a persistent disk: each new image build must include the source DEM again.
- This free setup is suitable for a demonstration, not a guarantee of continuous 24/7 availability. The project has not been deployed or verified on Render by this repository change.

## Before creating services

1. Confirm that the repository's `master` branch contains `render.yaml`, `render.Dockerfile`, `render.Dockerfile.dockerignore`, and the backend DEM and GeoJSON at the paths named in the Dockerfile.
2. Keep a separate backup of the original DEM. The deployment image is not a backup.
3. The expected Render subdomains are:
   - Frontend: `https://mars-pathfinder-frontend.onrender.com`
   - API: `https://mars-pathfinder-api.onrender.com`
4. If Render assigns different subdomains because either name is already taken, update `API_ALLOWED_ORIGINS` on the API service and `VITE_API_BASE_URL` on the static site to match the actual URLs. The frontend URL is embedded during the static-site build, so redeploy the static site after changing it.

## Connect GitHub and create the Blueprint

1. Sign in to Render and connect the GitHub account that can access `Mars-Pathfinder-2026`.
2. Choose **New → Blueprint** and select the existing repository and the `master` branch.
3. Review the two services described in `render.yaml`:
   - `mars-pathfinder-frontend`: free static site, built with `npm ci` and `npm run build`.
   - `mars-pathfinder-api`: free Docker web service, built from the Render-specific Dockerfile.
4. Confirm both compute choices are **Free**. Do not attach a disk or select a paid plan for this configuration.
5. Create the Blueprint. This repository does not contain credentials or API keys, and no secrets are required for the current application.

## Check the deployment

After both services report a successful deploy, open the frontend URL in a browser and check the map. Then check these API URLs directly:

- `https://mars-pathfinder-api.onrender.com/api/health` — expected JSON: `{"status":"ok"}`
- `https://mars-pathfinder-api.onrender.com/api/sites` — expected site collection

Use the map to place a landing point and request a terrain assessment. Then choose a science site and plan routes. For a direct route request, send JSON such as:

```json
{
  "start_lon": 77.7802176586129,
  "start_lat": 18.446480857911807,
  "target_site": "Montagnac"
}
```

to `POST /api/plan-routes`. For landing assessment, send the same coordinate pair as `longitude` and `latitude` to `POST /api/landing-point`.

The API allows browser requests only from the configured frontend origin. CORS is a browser policy, not authentication; the API endpoints remain publicly reachable.

## Automatic updates

The Blueprint sets both services to deploy automatically for commits to `master`. After you later push an approved update to GitHub, Render builds the affected service from that branch. This local preparation did not create a commit or push any changes.

## DEM and other data

The backend Docker build copies the DEM and site GeoJSON into the image. Render's Free filesystem is ephemeral for runtime writes, so do not store new persistent user data there. For future changes to scientific inputs, update the repository source through your normal review process and redeploy; keep an independent backup of the original NASA/USGS files.

## Troubleshooting

- **Map opens but sites do not load:** verify `VITE_API_BASE_URL` points to the API service and that the API's `API_ALLOWED_ORIGINS` exactly matches the frontend origin. Rebuild the static site after changing either value.
- **API health check fails:** inspect the Render API service logs and confirm the Docker build had access to both data files. The health endpoint does not run a terrain analysis.
- **First request is slow:** the Free API may have been spun down. Wait for it to start and retry.
- **Landing assessment or route request times out:** Render Free has limited CPU and memory. A locally successful operation does not guarantee cloud latency or capacity.

## Scientific scope

Terrain and route outputs are planning estimates derived from the available DEM and current project methods. They are not guarantees of landing or traversal safety. Mars longitude/latitude interpretation remains an assumption documented by the engine.
