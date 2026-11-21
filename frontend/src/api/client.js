export function buildScanWsUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/scan`
}

async function parseJsonResponse(response) {
  const text = await response.text()
  let data = {}
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = { error: text }
    }
  }
  if (!response.ok) {
    const message = data.error || data.detail || `Request failed with status ${response.status}`
    throw new Error(message)
  }
  if (data && data.error) {
    throw new Error(data.error)
  }
  return data
}

export async function fetchHealthStatus() {
  const tStart = performance.now()
  const res = await fetch('/api/v1/health')
  const data = await parseJsonResponse(res)
  const latencyMs = Math.round(performance.now() - tStart)
  return {
    data,
    latencyMs,
  }
}

export async function detectImageFile({
  file,
  detectorMode,
  enableClahe,
  confThreshold,
  sessionId,
  excludedClasses,
}) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('detector_mode', detectorMode)
  fd.append('enable_clahe', enableClahe ? 'true' : 'false')
  fd.append('conf_threshold', String(confThreshold))
  if (sessionId) {
    fd.append('session_id', sessionId)
  }
  if (excludedClasses && excludedClasses.length > 0) {
    fd.append('excluded_classes', JSON.stringify(excludedClasses))
  }
  const res = await fetch('/api/v1/ui/detect', {
    method: 'POST',
    body: fd,
  })
  return parseJsonResponse(res)
}

export async function detectSampleImage({
  sampleName,
  detectorMode,
  enableClahe,
  confThreshold,
  sessionId,
  excludedClasses,
}) {
  const res = await fetch('/api/v1/ui/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sample_name: sampleName,
      detector_mode: detectorMode,
      enable_clahe: Boolean(enableClahe),
      conf_threshold: Number(confThreshold),
      session_id: sessionId,
      excluded_classes: excludedClasses || [],
    }),
  })
  return parseJsonResponse(res)
}

export async function detectSnapshotBase64({
  imageB64,
  detectorMode,
  enableClahe,
  confThreshold,
  sessionId,
  excludedClasses,
}) {
  const res = await fetch('/api/v1/ui/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_b64: imageB64,
      detector_mode: detectorMode,
      enable_clahe: Boolean(enableClahe),
      conf_threshold: Number(confThreshold),
      session_id: sessionId,
      excluded_classes: excludedClasses || [],
    }),
  })
  return parseJsonResponse(res)
}

export async function scanVideoClip({
  file,
  detectorMode,
  enableClahe,
  confThreshold,
}) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('detector_mode', detectorMode)
  fd.append('enable_clahe', enableClahe ? 'true' : 'false')
  fd.append('conf_threshold', String(confThreshold))
  const res = await fetch('/api/v1/scan/video', {
    method: 'POST',
    body: fd,
  })
  return parseJsonResponse(res)
}

export async function generateRecipesFromBasket(items, timeoutSec = 8.0) {
  const payloadItems = items.map(function mapBasketItem(item) {
    const count = Math.max(1, Number(item.count || 1))
    const mapped = {
      class_id: item.class_id,
      display_name: item.display_name,
      count,
      storage_zone: String(item.storage_zone || 'pantry').toLowerCase(),
      crop_jpeg_b64: item.crop_jpeg_b64 || null,
    }
    if (Number(item.calories_per_unit) > 0) {
      mapped.calories_per_unit = Math.round(Number(item.calories_per_unit))
    }
    if (Number(item.estimated_calories) > 0) {
      mapped.estimated_calories = Math.round(Number(item.estimated_calories))
    }
    return mapped
  })
  const res = await fetch('/api/v1/recipes/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      confirmed_inventory: payloadItems,
      timeout_sec: timeoutSec,
    }),
  })
  return parseJsonResponse(res)
}

export async function resetBackendSession(sessionId) {
  const res = await fetch('/api/v1/session/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId || 'default',
    }),
  })
  return parseJsonResponse(res)
}

export async function deleteSessionItemRest(sessionId, classId) {
  const res = await fetch('/api/v1/session/items/delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId || 'default',
      class_id: classId,
    }),
  })
  return parseJsonResponse(res)
}

export async function restoreSessionItemRest(sessionId, classId) {
  const res = await fetch('/api/v1/session/items/restore', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId || 'default',
      class_id: classId,
    }),
  })
  return parseJsonResponse(res)
}
