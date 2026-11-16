
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')
const apiUrl = (path) => `${API_BASE_URL}${path}`

delete L.Icon.Default.prototype._getIconUrl

L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
})

const map = L.map('map', {
  crs: L.CRS.Simple,
  minZoom: -4,
  maxZoom: 5,
  zoomSnap: 0.25,
  attributionControl: false,
  maxBoundsViscosity: 1.0,
})

const panel = document.getElementById('panel')
let info = null
let landing = null
let landingAssessment = null
let landingAssessmentStatus = 'idle'
let landingAssessmentError = ''
let landingAssessmentSequence = 0

let destination = null
let destLayer = null

let result = null
let status = 'idle'
let errorMsg = ''
let selected = 0

const routeGroup = L.layerGroup().addTo(map)
let routeLines = []

const ROUTE_STYLES = [
  { color: '#4dff88', dashArray: null },
  { color: '#ff4dd2', dashArray: '8 6' },
]

const esc = (s) =>
  String(s ?? '').replace(/[&<>"]/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;'
  }[c]))

const fmt = (v, d = 2) =>
  typeof v === 'number' && isFinite(v) ? v.toFixed(d) : 'not available'

function landingAssessmentHtml() {
  if (landingAssessmentStatus === 'loading') {
    return '<p class="hint">Assessing slope and surface roughness around this landing point...</p>'
  }

  if (landingAssessmentStatus === 'error') {
    return `<p class="hint">Terrain assessment unavailable: ${esc(landingAssessmentError)}</p>`
  }

  if (landingAssessmentStatus !== 'done' || !landingAssessment?.assessment) {
    return ''
  }

  const a = landingAssessment.assessment
  const p = a.class_percentages || {}

  const percent = (v) =>
    typeof v === 'number' && isFinite(v) ? `${v.toFixed(1)}%` : 'not available'

  return `
    <h3>Terrain assessment (500 m radius)</h3>
    <dl>
      <dt>Estimated suitability</dt><dd>${esc(a.final_assessment ?? 'not available')}</dd>
      <dt>More Suitable</dt><dd>${percent(p['More Suitable'])}</dd>
      <dt>Caution</dt><dd>${percent(p.Caution)}</dd>
      <dt>Unsuitable</dt><dd>${percent(p.Unsuitable)}</dd>
      <dt>Mean slope</dt><dd>${fmt(a.mean_slope_deg)}°</dd>
      <dt>Mean roughness</dt><dd>${fmt(a.mean_roughness_m)} m</dd>
      <dt>Valid pixels</dt><dd>${a.valid_pixel_count ?? 'not available'}</dd>
    </dl>
    <p class="hint">${esc(a.assessment_rule ?? '')}</p>
    <p class="hint">Preliminary terrain estimate only; it does not guarantee landing safety.</p>
  `
}

async function assessLandingPoint(point) {
  const sequence = ++landingAssessmentSequence

  landingAssessment = null
  landingAssessmentStatus = 'loading'
  landingAssessmentError = ''
  renderPanel()

  try {
    const response = await fetch(apiUrl('/api/landing-point'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        longitude: point.lon,
        latitude: point.lat,
        radius_m: 500,
      }),
    })

    const data = await response.json()

    if (!response.ok || data.success !== true) {
      throw new Error(
        data.error?.message || `The server returned HTTP ${response.status}.`
      )
    }

    if (sequence !== landingAssessmentSequence) return

    landingAssessment = data
    landingAssessmentStatus = 'done'
  } catch (error) {
    if (sequence !== landingAssessmentSequence) return

    landingAssessmentStatus = 'error'
    landingAssessmentError =
      error.message || 'Could not reach the terrain assessment API.'
  }

  renderPanel()
}

// Leaflet pixel -> Mars lon/lat
function toMars(latlng) {
  const col = latlng.lng
  const row = info.height - latlng.lat
  const x = info.x0 + col * info.pixel
  const y = info.y0 - row * info.pixel
  const R = info.radius
  const lon = (x / (R * Math.cos((info.lat0 * Math.PI) / 180))) * (180 / Math.PI)
  const lat = (y / R) * (180 / Math.PI)
  return { lon, lat, row, col }
}

// Mars lon/lat -> Leaflet pixel
function fromMars(lon, lat) {
  const R = info.radius
  const x = R * ((lon * Math.PI) / 180) * Math.cos((info.lat0 * Math.PI) / 180)
  const y = R * ((lat * Math.PI) / 180)
  const col = (x - info.x0) / info.pixel
  const row = (info.y0 - y) / info.pixel
  return L.latLng(info.height - row, col)
}

function routeCard(r, i) {
  const st = ROUTE_STYLES[i] || ROUTE_STYLES[ROUTE_STYLES.length - 1]
  const d = r.distance_m
  const dist =
    typeof d === 'number' && isFinite(d) ? `${(d / 1000).toFixed(2)} km (${Math.round(d)} m)` : 'not available'
  const cost =
    typeof r.weighted_cost_m === 'number' && isFinite(r.weighted_cost_m)
      ? `${Math.round(r.weighted_cost_m)} m-equivalent`
      : 'not available'
  const bg = i === selected ? 'rgba(255,255,255,0.08)' : 'transparent'

  return `
    <div data-route="${i}" style="border-left:4px solid ${st.color}; padding:4px 8px; margin:10px 0; cursor:pointer; background:${bg}">
      <strong>${esc(r.route_id ?? 'Route ' + (i + 1))}</strong>
      <dl>
        <dt>Distance (actual path length)</dt><dd>${dist}</dd>
        <dt>Mean slope</dt><dd>${fmt(r.mean_slope_deg)}°</dd>
        <dt>Mean roughness</dt><dd>${fmt(r.mean_roughness_m)} m</dd>
        <dt>Weighted cost (not a distance)</dt><dd>${cost}</dd>
        <dt>Caution cells</dt><dd>${r.caution_cells ?? 'not available'}</dd>
        <dt>Unsuitable cells</dt><dd>${r.unsuitable_cells ?? 'not available'}</dd>
      </dl>
    </div>`
}

function routesHtml() {
  const ready = landing && destination
  const busy = status === 'loading'
  let h = '<h2>Routes</h2>'

  h += `<button id="plan-btn" ${!ready || busy ? 'disabled' : ''}
    style="width:100%; padding:10px; font-weight:700; border:0; cursor:pointer; background:#c1440e; color:#fff; opacity:${!ready || busy ? 0.5 : 1}">
    ${busy ? 'Asking the route engine...' : 'Plan routes'}</button>`

  if (!ready) h += '<p class="hint">Choose a landing point and a destination first.</p>'
  if (status === 'error') h += `<p class="hint">⚠ ${esc(errorMsg)}</p>`

  if (status === 'done' && result) {
    const routes = result.routes || []

    if (!routes.length) h += '<p class="hint">The engine returned no routes.</p>'

    routes.forEach((r, i) => (h += routeCard(r, i)))

    h += `<p class="hint">Alternative route: ${result.alternative_available ? 'available' : 'not available'}.
      ${esc(result.alternative_message ?? '')}</p>`

    const a = result.assumptions

    if (Array.isArray(a) && a.length) {
      h += `<p class="hint">Engine assumptions:</p><ul>${a.map((x) => `<li class="hint">${esc(x)}</li>`).join('')}</ul>`
    } else if (typeof a === 'string' && a) {
      h += `<p class="hint">Engine assumptions: ${esc(a)}</p>`
    }
  }

  return h
}

function renderPanel() {
  let html = '<h2>Landing Site</h2>'

  if (!landing) {
    html += '<p class="hint">Click on the map to choose a landing point.</p>'
  } else {
    html += `
      <dl>
        <dt>Longitude (°E)</dt><dd>${landing.lon.toFixed(5)}</dd>
        <dt>Latitude (°N)</dt><dd>${landing.lat.toFixed(5)}</dd>
        <dt>DEM row / col</dt><dd>${Math.floor(landing.row)} / ${Math.floor(landing.col)}</dd>
      </dl>
      ${landingAssessmentHtml()}`
  }

  html += '<h2>Destination</h2>'

  if (!destination) {
    html += '<p class="hint">Click a science site on the map to choose it as the destination.</p>'
  } else {
    const p = destination.properties
    const [lon, lat] = destination.geometry.coordinates
    const elev = p.elevation_m == null ? 'not available' : `${p.elevation_m} m`

    const warn =
      p.coordinate_confidence !== 'High'
        ? `<p class="hint">⚠ Coordinate confidence: ${esc(p.coordinate_confidence)}. See rationale below.</p>`
        : ''

    html += `
      <dl>
        <dt>Site</dt><dd>${esc(p.site_name)}</dd>
        <dt>Geologic unit</dt><dd>${esc(p.geologic_unit)}</dd>
        <dt>Coordinates (lon, lat)</dt><dd>${lon}, ${lat}</dd>
        <dt>Elevation</dt><dd>${esc(elev)}</dd>
        <dt>Coordinate confidence</dt><dd>${esc(p.coordinate_confidence)}</dd>
      </dl>
      ${warn}
      <p class="hint">${esc(p.scientific_rationale)}</p>
      <p class="hint">Source: ${esc(p.source)}</p>`
  }

  html += routesHtml()
  html += '<p class="hint">Note: the science-site reference is not yet fully verified scientifically.</p>'
  panel.innerHTML = html
}

function clearRoutes() {
  routeGroup.clearLayers()
  routeLines = []
  result = null
  status = 'idle'
  errorMsg = ''
  selected = 0
}

function selectRoute(i) {
  selected = i
  routeLines.forEach((l, j) => l.setStyle({ weight: j === i ? 7 : 4 }))
  if (routeLines[i]) routeLines[i].bringToFront()
  renderPanel()
}

function drawRoutes(routes) {
  const all = []

  routes.forEach((r, i) => {
    const ok =
      Array.isArray(r.coordinates) &&
      r.coordinates.length > 1 &&
      r.coordinates.every((c) => Array.isArray(c) && isFinite(c[0]) && isFinite(c[1]))

    if (!ok) throw new Error('Route coordinates are not in the expected [lon, lat] format.')

    const pts = r.coordinates.map(([lon, lat]) => fromMars(lon, lat))
    all.push(...pts)

    const st = ROUTE_STYLES[i] || ROUTE_STYLES[ROUTE_STYLES.length - 1]

    const line = L.polyline(pts, {
      color: st.color,
      dashArray: st.dashArray,
      weight: i === 0 ? 7 : 4,
      opacity: 0.95,
      bubblingMouseEvents: false,
    }).addTo(routeGroup)

    line.on('click', () => selectRoute(i))
    routeLines.push(line)
  })

  if (all.length) map.fitBounds(L.latLngBounds(all), { padding: [60, 60] })
}

function describeError(statusCode, data) {
  if ([502, 503, 504].includes(statusCode)) {
    return 'Could not reach the route engine. Is the backend running on port 8000?'
  }

  const d = data && data.detail

  if (typeof d === 'string') return d
  if (d && typeof d.message === 'string') return d.message
  if (d) return JSON.stringify(d)

  return `The server returned an error (HTTP ${statusCode}).`
}

async function planRoutes() {
  if (!landing || !destination) return

  clearRoutes()
  status = 'loading'
  renderPanel()

  try {
    const res = await fetch(apiUrl('/api/plan-routes'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start_lon: landing.lon,
        start_lat: landing.lat,
        target_site: destination.properties.site_name,
      }),
    })

    let data = null

    try {
      data = await res.json()
    } catch (_) {
      /* response was not JSON */
    }

    if (!data || data.success !== true) {
      const raw = data && (data.message || data.error || data.detail)

      const text =
        typeof raw === 'string'
          ? raw
          : raw && typeof raw.message === 'string'
            ? raw.message
            : raw
              ? JSON.stringify(raw)
              : ''

      throw new Error(text || 'The engine reported that it could not plan a route.')
    }

    result = data
    drawRoutes(data.routes || [])
    status = 'done'
  } catch (err) {
    routeGroup.clearLayers()
    routeLines = []
    result = null
    status = 'error'

    errorMsg =
      err instanceof TypeError
        ? 'Could not reach the API. Please try again when the service is available.'
        : err.message
  }

  renderPanel()
}

function selectSite(feature, layer) {
  clearRoutes()
  destination = feature

  if (destLayer) destLayer.setStyle({ weight: 2, radius: 7 })

  destLayer = layer
  layer.setStyle({ weight: 4, radius: 10 })

  renderPanel()
}

panel.addEventListener('click', (e) => {
  if (e.target.closest('#plan-btn')) {
    planRoutes()
    return
  }

  const card = e.target.closest('[data-route]')
  if (card) selectRoute(Number(card.dataset.route))
})

async function init() {
  const res = await fetch('/dem_info.json')
  info = await res.json()

  const bounds = [[0, 0], [info.height, info.width]]

  L.imageOverlay('/jezero_basemap.png', bounds).addTo(map)
  map.fitBounds(bounds)

  const sitesRes = await fetch(apiUrl('/api/sites'))
  if (!sitesRes.ok) throw new Error(`/api/sites returned ${sitesRes.status}`)

  const { sites } = await sitesRes.json()
  const points = []

  sites.forEach((feature) => {
    const [lon, lat] = feature.geometry.coordinates
    const pos = fromMars(lon, lat)

    points.push(pos)

    const high = feature.properties.coordinate_confidence === 'High'

    const layer = L.circleMarker(pos, {
      radius: 7,
      weight: 2,
      color: '#ffffff',
      fillColor: high ? '#ffb347' : '#00e5ff',
      fillOpacity: 0.95,
      dashArray: high ? null : '3 3',
      bubblingMouseEvents: false,
    }).addTo(map)

    layer.bindTooltip(feature.properties.site_name)
    layer.on('click', () => selectSite(feature, layer))
  })

  // if (points.length) map.fitBounds(L.latLngBounds(points), { padding: [80, 80] })

  map.on('click', (e) => {
    clearRoutes()

    if (landing) map.removeLayer(landing.marker)

    landing = { marker: L.marker(e.latlng).addTo(map), ...toMars(e.latlng) }

    landingAssessment = null
    landingAssessmentStatus = 'loading'
    landingAssessmentError = ''

    renderPanel()
    assessLandingPoint(landing)
  })

  renderPanel()
}

init().catch((err) => {
  panel.innerHTML = `<h2>Error</h2><p class="hint">Could not load map data: ${esc(err.message)}</p>`
})
