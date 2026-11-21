import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ChefHat,
  ScanLine,
  ShoppingBasket,
} from 'lucide-react'
import {
  deleteSessionItemRest,
  fetchHealthStatus,
  generateRecipesFromBasket,
  resetBackendSession,
  restoreSessionItemRest,
} from './api/client.js'
import BasketView from './components/BasketView.jsx'
import DiagnosticsModal from './components/DiagnosticsModal.jsx'
import HeaderBar from './components/HeaderBar.jsx'
import RecipesView from './components/RecipesView.jsx'
import ScannerView from './components/ScannerView.jsx'
import {
  DETECTOR_MODES,
  buildBasketItemKey,
  computeTotalBasketCalories,
  createClientSessionId,
  getCaloriesPerUnit,
  getClassMeta,
  normalizeClassId,
} from './constants/catalog.js'

const STORAGE_KEY_BASKET = 'stock2stove.basket_items'
const STORAGE_KEY_EXCLUDED = 'stock2stove.excluded_classes'
const STORAGE_KEY_PREFS = 'stock2stove.preferences'
const VALID_TABS = new Set(['scanner', 'basket', 'recipes'])

function readTabFromHash() {
  try {
    const raw = String(window.location.hash || '').replace(/^#/, '').trim().toLowerCase()
    if (VALID_TABS.has(raw)) {
      return raw
    }
  } catch {
  }
  return 'scanner'
}

function doesBasketItemMatchTarget(item, targetKey) {
  const rowKey = item.item_key || buildBasketItemKey(item.class_id, item.display_name)
  if (rowKey === targetKey) {
    return true
  }
  const normTarget = normalizeClassId(targetKey)
  if (normTarget && normalizeClassId(item.class_id) === normTarget) {
    return true
  }
  return false
}

function readStoredJson(key, fallbackValue) {
  try {
    const raw = window.localStorage.getItem(key)
    if (!raw) {
      return fallbackValue
    }
    const parsed = JSON.parse(raw)
    return parsed == null ? fallbackValue : parsed
  } catch {
    return fallbackValue
  }
}

function writeStoredJson(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch {
  }
}

function writeStoredBasket(items) {
  try {
    window.localStorage.setItem(STORAGE_KEY_BASKET, JSON.stringify(items))
  } catch {
    try {
      const compactItems = items.map(function stripCrop(item) {
        return {
          ...item,
          crop_jpeg_b64: null,
        }
      })
      window.localStorage.setItem(STORAGE_KEY_BASKET, JSON.stringify(compactItems))
    } catch {
    }
  }
}

function removeStoredKey(key) {
  try {
    window.localStorage.removeItem(key)
  } catch {
  }
}

function loadInitialPreferences() {
  const validModes = new Set(DETECTOR_MODES.map(function getModeVal(m) { return m.value }))
  const raw = readStoredJson(STORAGE_KEY_PREFS, {})
  const prefs = raw && typeof raw === 'object' ? raw : {}

  const rawMode = typeof prefs.detectorMode === 'string' ? prefs.detectorMode : 'gemini_vision'
  const detectorMode = validModes.has(rawMode) ? rawMode : 'gemini_vision'
  const enableClahe = typeof prefs.enableClahe === 'boolean' ? prefs.enableClahe : true
  const numConf = Number(prefs.confThreshold)
  const confThreshold = Number.isFinite(numConf) ? Math.max(0.2, Math.min(0.85, numConf)) : 0.45
  const facingMode = prefs.facingMode === 'user' || prefs.facingMode === 'environment'
    ? prefs.facingMode
    : 'environment'

  return {
    detectorMode,
    enableClahe,
    confThreshold,
    facingMode,
  }
}

function loadInitialExcludedClasses() {
  const raw = readStoredJson(STORAGE_KEY_EXCLUDED, [])
  if (!Array.isArray(raw)) {
    return []
  }
  const seen = new Set()
  const out = []
  for (let i = 0; i < raw.length; i += 1) {
    const norm = normalizeClassId(raw[i])
    if (norm && !seen.has(norm)) {
      seen.add(norm)
      out.push(norm)
    }
  }
  return out
}

function resolveItemCalories(item, count, fallbackPerUnit) {
  const rawPerUnit = Number(item.calories_per_unit)
  const rawEst = Number(item.estimated_calories)
  if (Number.isFinite(rawPerUnit) && rawPerUnit > 0) {
    const perUnit = Math.round(rawPerUnit)
    const estCalories = Number.isFinite(rawEst) && rawEst > 0 ? Math.round(rawEst) : Math.round(perUnit * count)
    return { perUnit, estCalories, hasExplicit: true }
  }
  if (Number.isFinite(rawEst) && rawEst > 0) {
    const estCalories = Math.round(rawEst)
    const perUnit = Math.max(1, Math.round(estCalories / count))
    return { perUnit, estCalories, hasExplicit: true }
  }
  const perUnit = Math.round(fallbackPerUnit)
  return { perUnit, estCalories: Math.round(perUnit * count), hasExplicit: false }
}

function loadInitialBasketItems(excludedList) {
  const raw = readStoredJson(STORAGE_KEY_BASKET, [])
  if (!Array.isArray(raw)) {
    return []
  }
  const excludedSet = new Set(excludedList.map(normalizeClassId))
  const out = []
  for (let i = 0; i < raw.length; i += 1) {
    const row = raw[i]
    if (!row || typeof row !== 'object') {
      continue
    }
    const normId = normalizeClassId(row.class_id)
    if (!normId || excludedSet.has(normId)) {
      continue
    }
    const meta = getClassMeta(normId)
    const displayName = row.display_name || meta.display_name
    const itemKey = row.item_key || buildBasketItemKey(normId, displayName)
    const count = Math.max(1, Number(row.count || 1))
    const fallbackPerUnit = Number(meta.calories_per_unit || getCaloriesPerUnit(normId))
    const resolved = resolveItemCalories(row, count, fallbackPerUnit)
    out.push({
      item_key: itemKey,
      class_id: normId,
      display_name: displayName,
      storage_zone: row.storage_zone || meta.storage_zone,
      count,
      max_conf: Number(row.max_conf || 0),
      crop_jpeg_b64: row.crop_jpeg_b64 || null,
      calories_per_unit: resolved.perUnit,
      estimated_calories: resolved.estCalories,
    })
  }
  return out
}

function mergeBasketInventory(prevItems, incomingItems, excludedList) {
  const excludedSet = new Set(excludedList.map(normalizeClassId))
  const byKey = new Map()

  for (let i = 0; i < prevItems.length; i += 1) {
    const prev = prevItems[i]
    const normId = normalizeClassId(prev.class_id)
    if (!normId || excludedSet.has(normId)) {
      continue
    }
    const meta = getClassMeta(normId)
    const prevDisp = prev.display_name || meta.display_name
    const prevKey = prev.item_key || buildBasketItemKey(normId, prevDisp)
    byKey.set(prevKey, {
      ...prev,
      item_key: prevKey,
      class_id: normId,
      display_name: prevDisp,
    })
  }

  for (let i = 0; i < incomingItems.length; i += 1) {
    const inc = incomingItems[i]
    const normId = normalizeClassId(inc.class_id)
    if (!normId || excludedSet.has(normId)) {
      continue
    }
    const meta = getClassMeta(normId)
    const incDisp = inc.display_name || meta.display_name
    const itemKey = inc.item_key || buildBasketItemKey(normId, incDisp)
    const incCount = Math.max(1, Number(inc.count || 1))
    const incConf = Number(inc.max_conf || 0)
    const existing = byKey.get(itemKey)
    const defaultPerUnit = Number(meta.calories_per_unit || getCaloriesPerUnit(normId))
    const existingPerUnit = existing && Number(existing.calories_per_unit) > 0
      ? Number(existing.calories_per_unit)
      : defaultPerUnit
    const resolved = resolveItemCalories(inc, incCount, existingPerUnit)

    if (existing) {
      const nextCount = Math.max(Number(existing.count || 1), incCount)
      const nextConf = Math.max(Number(existing.max_conf || 0), incConf)
      const nextCrop = inc.crop_jpeg_b64 && (incConf >= Number(existing.max_conf || 0) || !existing.crop_jpeg_b64)
        ? inc.crop_jpeg_b64
        : existing.crop_jpeg_b64
      let nextEst = Math.round(resolved.perUnit * nextCount)
      if (resolved.hasExplicit && nextCount === incCount) {
        nextEst = resolved.estCalories
      } else if (!resolved.hasExplicit && nextCount === Number(existing.count || 1) && Number(existing.estimated_calories) > 0) {
        nextEst = Math.round(Number(existing.estimated_calories))
      }
      byKey.set(itemKey, {
        item_key: itemKey,
        class_id: normId,
        display_name: inc.display_name || existing.display_name || meta.display_name,
        storage_zone: existing.storage_zone || meta.storage_zone,
        count: nextCount,
        max_conf: nextConf,
        crop_jpeg_b64: nextCrop || null,
        calories_per_unit: resolved.perUnit,
        estimated_calories: nextEst,
      })
    } else {
      byKey.set(itemKey, {
        item_key: itemKey,
        class_id: normId,
        display_name: incDisp,
        storage_zone: meta.storage_zone,
        count: incCount,
        max_conf: incConf,
        crop_jpeg_b64: inc.crop_jpeg_b64 || null,
        calories_per_unit: resolved.perUnit,
        estimated_calories: resolved.estCalories,
      })
    }
  }

  return Array.from(byKey.values())
}

export default function App() {
  const [initialState] = useState(function initPersistedState() {
    const prefs = loadInitialPreferences()
    const excluded = loadInitialExcludedClasses()
    const basket = loadInitialBasketItems(excluded)
    return { prefs, excluded, basket }
  })

  const [activeTab, setActiveTab] = useState(readTabFromHash)
  const [sessionId] = useState(createClientSessionId)
  const [detectorMode, setDetectorMode] = useState(initialState.prefs.detectorMode)
  const [enableClahe, setEnableClahe] = useState(initialState.prefs.enableClahe)
  const [confThreshold, setConfThreshold] = useState(initialState.prefs.confThreshold)
  const [facingMode, setFacingMode] = useState(initialState.prefs.facingMode)

  const [basketItems, setBasketItems] = useState(initialState.basket)
  const [excludedClasses, setExcludedClasses] = useState(initialState.excluded)
  const excludedRef = useRef(initialState.excluded)
  const wsSendRef = useRef(null)

  const [recipeState, setRecipeState] = useState({
    status: 'idle',
    data: null,
    error: '',
  })
  const [checkedSteps, setCheckedSteps] = useState({})

  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false)
  const [healthState, setHealthState] = useState({
    status: 'checking',
    data: null,
    latencyMs: null,
    error: '',
  })
  const [logs, setLogs] = useState([])

  const appendLog = useCallback(function logFn(level, tag, message) {
    const now = new Date()
    const timeStr = now.toTimeString().slice(0, 8)
    const entry = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2, 7)}`,
      level: String(level || 'INFO').toUpperCase(),
      tag: String(tag || 'APP'),
      message: String(message || ''),
      time: timeStr,
    }
    setLogs(function prependLog(prev) {
      return [entry].concat(prev).slice(0, 120)
    })
  }, [])

  const refreshHealth = useCallback(async function loadHealth() {
    setHealthState(function markChecking(prev) {
      return { ...prev, status: 'checking', error: '' }
    })
    try {
      const result = await fetchHealthStatus()
      setHealthState({
        status: 'connected',
        data: result.data,
        latencyMs: result.latencyMs,
        error: '',
      })
      appendLog('INFO', 'HEALTH', `Server healthy (${result.latencyMs} ms)`)
    } catch (err) {
      const msg = err.message || 'Failed to connect to backend server'
      setHealthState(function markErr(prev) {
        return {
          ...prev,
          status: 'error',
          error: msg,
        }
      })
      appendLog('ERROR', 'HEALTH', msg)
    }
  }, [appendLog])

  useEffect(function initHealthCheck() {
    refreshHealth()
  }, [refreshHealth])

  useEffect(function initHistoryEffect() {
    const initialTab = readTabFromHash()
    try {
      window.history.replaceState({ tab: initialTab, diagnosticsOpen: false }, '', `#${initialTab}`)
    } catch {
    }
    function handlePopState(event) {
      const state = event && event.state ? event.state : null
      const nextTab = state && VALID_TABS.has(state.tab) ? state.tab : readTabFromHash()
      const nextDiag = Boolean(state && state.diagnosticsOpen)
      setActiveTab(nextTab)
      setDiagnosticsOpen(nextDiag)
    }
    window.addEventListener('popstate', handlePopState)
    return function cleanupPopState() {
      window.removeEventListener('popstate', handlePopState)
    }
  }, [])

  useEffect(function persistBasketEffect() {
    writeStoredBasket(basketItems)
  }, [basketItems])

  useEffect(function persistExcludedEffect() {
    writeStoredJson(STORAGE_KEY_EXCLUDED, excludedClasses)
  }, [excludedClasses])

  useEffect(function persistPrefsEffect() {
    writeStoredJson(STORAGE_KEY_PREFS, {
      detectorMode,
      enableClahe,
      confThreshold,
      facingMode,
    })
  }, [detectorMode, enableClahe, confThreshold, facingMode])

  function registerWsSend(fn) {
    wsSendRef.current = fn
  }

  function navigateToTab(nextTab) {
    const targetTab = VALID_TABS.has(nextTab) ? nextTab : 'scanner'
    if (targetTab === activeTab) {
      if (diagnosticsOpen) {
        handleCloseDiagnostics()
      }
      return
    }
    setDiagnosticsOpen(false)
    setActiveTab(targetTab)
    try {
      window.history.pushState({ tab: targetTab, diagnosticsOpen: false }, '', `#${targetTab}`)
    } catch {
    }
  }

  function handleScanUpdate({ confirmed_inventory, excluded_classes, recipes, replaceInventory, clearOnly }) {
    const serverExcluded = Array.isArray(excluded_classes)
      ? excluded_classes.map(normalizeClassId).filter(Boolean)
      : []

    let combinedExcluded = excludedRef.current
    if (serverExcluded.length > 0) {
      const mergedExc = new Set(excludedRef.current.concat(serverExcluded))
      combinedExcluded = Array.from(mergedExc)
      excludedRef.current = combinedExcluded
      setExcludedClasses(combinedExcluded)
    }

    if (clearOnly) {
      const excSet = new Set(combinedExcluded)
      setBasketItems(function filterOnly(prev) {
        return prev.filter(function keepRow(row) {
          return !excSet.has(normalizeClassId(row.class_id))
        })
      })
      return
    }

    const hasRecipes = Boolean(recipes && Array.isArray(recipes.recipes) && recipes.recipes.length > 0)

    if (replaceInventory && Array.isArray(confirmed_inventory)) {
      setBasketItems(mergeBasketInventory([], confirmed_inventory, combinedExcluded))
      if (!hasRecipes) {
        setRecipeState({ status: 'idle', data: null, error: '' })
      }
    } else if (Array.isArray(confirmed_inventory) && confirmed_inventory.length > 0) {
      setBasketItems(function mergeNext(prev) {
        return mergeBasketInventory(prev, confirmed_inventory, combinedExcluded)
      })
    }

    if (hasRecipes) {
      const usedFromConfirm = []
      if (Array.isArray(confirmed_inventory)) {
        for (let i = 0; i < confirmed_inventory.length; i += 1) {
          const cid = normalizeClassId(confirmed_inventory[i].class_id)
          if (cid) {
            usedFromConfirm.push(cid)
          }
        }
      }
      setRecipeState({
        status: 'success',
        data: {
          source: recipes.recipe_source || recipes.source || 'offline_cache',
          cached: Boolean(recipes.recipe_cached !== undefined ? recipes.recipe_cached : recipes.cached),
          latency_ms: recipes.inference_ms || 0,
          inventory_used: usedFromConfirm,
          items_used: recipes.items_used || null,
          total_inventory_calories: recipes.total_inventory_calories || recipes.total_estimated_calories || 0,
          recipes: recipes.recipes,
        },
        error: '',
      })
    }
  }

  async function handleDeleteItem(targetKey, excludeClass = false) {
    const rawPrefix = String(targetKey || '').split('::')[0]
    let normClassId = normalizeClassId(rawPrefix)
    const remainingItems = []
    for (let i = 0; i < basketItems.length; i += 1) {
      const row = basketItems[i]
      if (doesBasketItemMatchTarget(row, targetKey)) {
        if (!normClassId) {
          normClassId = normalizeClassId(row.class_id)
        }
      } else {
        remainingItems.push(row)
      }
    }
    if (!normClassId) {
      return
    }

    let hasRemainingInClass = false
    for (let i = 0; i < remainingItems.length; i += 1) {
      if (normalizeClassId(remainingItems[i].class_id) === normClassId) {
        hasRemainingInClass = true
        break
      }
    }
    const shouldExcludeClass = Boolean(excludeClass) || !hasRemainingInClass

    setBasketItems(function removeBasketRow(prev) {
      return prev.filter(function keepOther(item) {
        return !doesBasketItemMatchTarget(item, targetKey)
      })
    })
    setRecipeState({ status: 'idle', data: null, error: '' })

    if (!shouldExcludeClass) {
      appendLog('INFO', 'BASKET', `Removed ${targetKey} from basket`)
      return
    }

    const nextExcluded = Array.from(new Set(excludedRef.current.concat([normClassId])))
    excludedRef.current = nextExcluded
    setExcludedClasses(nextExcluded)
    appendLog('INFO', 'BASKET', `Removed ${normClassId} from basket and excluded from session`)

    if (wsSendRef.current) {
      wsSendRef.current({
        action: 'delete_item',
        class_id: normClassId,
      })
    }

    try {
      await deleteSessionItemRest(sessionId, normClassId)
    } catch (err) {
      appendLog('WARN', 'SESSION', `REST delete sync skipped: ${err.message}`)
    }
  }

  async function handleRestoreItem(rawClassId) {
    const normId = normalizeClassId(rawClassId)
    if (!normId) {
      return
    }
    const nextExcluded = excludedRef.current.filter(function dropId(c) { return c !== normId })
    excludedRef.current = nextExcluded
    setExcludedClasses(nextExcluded)
    appendLog('INFO', 'BASKET', `Restored ${normId} for detection`)

    if (wsSendRef.current) {
      wsSendRef.current({
        action: 'restore_item',
        class_id: normId,
      })
    }

    try {
      await restoreSessionItemRest(sessionId, normId)
    } catch (err) {
      appendLog('WARN', 'SESSION', `REST restore sync skipped: ${err.message}`)
    }
  }

  function handleIncrementItem(targetKey) {
    setBasketItems(function incRow(prev) {
      return prev.map(function updateRow(item) {
        if (!doesBasketItemMatchTarget(item, targetKey)) {
          return item
        }
        const normId = normalizeClassId(item.class_id)
        const nextCount = Number(item.count || 1) + 1
        const perUnit = Number(item.calories_per_unit || getCaloriesPerUnit(normId))
        return {
          ...item,
          count: nextCount,
          calories_per_unit: perUnit,
          estimated_calories: perUnit * nextCount,
        }
      })
    })
    setRecipeState({ status: 'idle', data: null, error: '' })
  }

  function handleDecrementItem(targetKey) {
    const target = basketItems.find(function findRow(item) {
      return doesBasketItemMatchTarget(item, targetKey)
    })
    if (target && Number(target.count || 1) <= 1) {
      handleDeleteItem(targetKey, false)
      return
    }
    setBasketItems(function decRow(prev) {
      return prev.map(function updateRow(item) {
        if (!doesBasketItemMatchTarget(item, targetKey)) {
          return item
        }
        const normId = normalizeClassId(item.class_id)
        const nextCount = Math.max(1, Number(item.count || 1) - 1)
        const perUnit = Number(item.calories_per_unit || getCaloriesPerUnit(normId))
        return {
          ...item,
          count: nextCount,
          calories_per_unit: perUnit,
          estimated_calories: perUnit * nextCount,
        }
      })
    })
    setRecipeState({ status: 'idle', data: null, error: '' })
  }

  function handleAddManualItem(classId, count) {
    const normId = normalizeClassId(classId)
    const qty = Math.max(1, Number(count || 1))
    const meta = getClassMeta(normId)
    const manualKey = buildBasketItemKey(normId, meta.display_name)
    const perUnit = meta.calories_per_unit || getCaloriesPerUnit(normId)

    if (excludedRef.current.includes(normId)) {
      handleRestoreItem(normId)
    }

    setBasketItems(function upsertManual(prev) {
      let matchIdx = -1
      for (let i = 0; i < prev.length; i += 1) {
        const rowKey = prev[i].item_key || buildBasketItemKey(prev[i].class_id, prev[i].display_name)
        if (rowKey === manualKey) {
          matchIdx = i
          break
        }
      }
      if (matchIdx === -1) {
        for (let i = 0; i < prev.length; i += 1) {
          if (normalizeClassId(prev[i].class_id) === normId) {
            matchIdx = i
            break
          }
        }
      }
      const updated = prev.slice()
      if (matchIdx >= 0) {
        const item = updated[matchIdx]
        const itemKey = item.item_key || buildBasketItemKey(item.class_id, item.display_name)
        const nextCount = Number(item.count || 1) + qty
        const rowPerUnit = Number(item.calories_per_unit) > 0 ? Number(item.calories_per_unit) : perUnit
        updated[matchIdx] = {
          ...item,
          item_key: itemKey,
          count: nextCount,
          calories_per_unit: rowPerUnit,
          estimated_calories: Math.round(rowPerUnit * nextCount),
        }
      } else {
        updated.push({
          item_key: manualKey,
          class_id: normId,
          display_name: meta.display_name,
          storage_zone: meta.storage_zone,
          count: qty,
          max_conf: 1.0,
          crop_jpeg_b64: null,
          calories_per_unit: perUnit,
          estimated_calories: perUnit * qty,
        })
      }
      return updated
    })
    setRecipeState({ status: 'idle', data: null, error: '' })
    appendLog('INFO', 'BASKET', `Manually added ${meta.display_name} x${qty}`)
  }

  async function handleResetSession() {
    excludedRef.current = []
    setExcludedClasses([])
    setBasketItems([])
    setRecipeState({ status: 'idle', data: null, error: '' })
    removeStoredKey(STORAGE_KEY_BASKET)
    removeStoredKey(STORAGE_KEY_EXCLUDED)
    appendLog('INFO', 'SESSION', `Reset session ${sessionId}`)
    try {
      await resetBackendSession(sessionId)
    } catch (err) {
      appendLog('WARN', 'SESSION', `Reset sync failed: ${err.message}`)
    }
  }

  async function handleClearBasket() {
    setRecipeState({ status: 'idle', data: null, error: '' })
    if (wsSendRef.current) {
      wsSendRef.current({
        frame_id: 0,
        detector_mode: detectorMode,
        enable_clahe: Boolean(enableClahe),
        conf_threshold: Number(confThreshold),
        image_b64: null,
        session_id: sessionId,
        reset_session: true,
      })
    }
    await handleResetSession()
  }

  async function handleGenerateRecipes() {
    if (basketItems.length === 0) {
      return
    }
    navigateToTab('recipes')
    setRecipeState({ status: 'loading', data: null, error: '' })
    try {
      const response = await generateRecipesFromBasket(basketItems, 30.0, true)
      setRecipeState({
        status: 'success',
        data: response,
        error: '',
      })
      appendLog(
        'INFO',
        'RECIPE',
        `Generated ${(response.recipes || []).length} recipes via ${response.source} (${response.total_inventory_calories || 0} kcal)`
      )
    } catch (err) {
      const msg = err.message || 'Failed to generate recipes'
      setRecipeState({
        status: 'error',
        data: null,
        error: msg,
      })
      appendLog('ERROR', 'RECIPE', msg)
    }
  }

  function handleToggleRecipeStep(recipeId, stepIdx) {
    setCheckedSteps(function toggleStep(prev) {
      const existing = prev[recipeId] || []
      const exists = existing.includes(stepIdx)
      const nextList = exists
        ? existing.filter(function drop(i) { return i !== stepIdx })
        : existing.concat([stepIdx])
      return {
        ...prev,
        [recipeId]: nextList,
      }
    })
  }

  function handleRemoveRecipe(recipeId) {
    setRecipeState(function removeRecipe(prev) {
      if (!prev.data || !Array.isArray(prev.data.recipes)) {
        return prev
      }
      const nextRecipes = prev.data.recipes.filter(function keepRecipe(recipe, idx) {
        const id = recipe.id || `recipe-${idx}`
        return id !== recipeId
      })
      if (nextRecipes.length === 0) {
        return {
          status: 'idle',
          data: null,
          error: '',
        }
      }
      return {
        ...prev,
        data: {
          ...prev.data,
          recipes: nextRecipes,
        },
      }
    })
    setCheckedSteps(function dropRecipeSteps(prev) {
      if (!Object.prototype.hasOwnProperty.call(prev, recipeId)) {
        return prev
      }
      const next = { ...prev }
      delete next[recipeId]
      return next
    })
    appendLog('INFO', 'RECIPE', `Removed recipe ${recipeId}`)
  }

  function handleOpenDiagnostics() {
    if (!diagnosticsOpen) {
      try {
        window.history.pushState({ tab: activeTab, diagnosticsOpen: true }, '', `#${activeTab}`)
      } catch {
      }
    }
    setDiagnosticsOpen(true)
    refreshHealth()
  }

  function handleCloseDiagnostics() {
    setDiagnosticsOpen(false)
    try {
      if (window.history.state && window.history.state.diagnosticsOpen) {
        window.history.back()
      }
    } catch {
    }
  }

  const basketTotalCalories = computeTotalBasketCalories(basketItems)
  const recipeCount = recipeState.data && Array.isArray(recipeState.data.recipes)
    ? recipeState.data.recipes.length
    : 0

  return (
    <div className="app-shell">
      <HeaderBar
        activeTab={activeTab}
        onSelectTab={navigateToTab}
        basketCount={basketItems.length}
        basketTotalCalories={basketTotalCalories}
        recipeCount={recipeCount}
        healthState={healthState}
        onOpenDiagnostics={handleOpenDiagnostics}
      />

      <main className="main-container">
        {activeTab === 'scanner' && (
          <ScannerView
            sessionId={sessionId}
            detectorMode={detectorMode}
            onChangeDetectorMode={setDetectorMode}
            enableClahe={enableClahe}
            onToggleClahe={setEnableClahe}
            confThreshold={confThreshold}
            onChangeConfThreshold={setConfThreshold}
            facingMode={facingMode}
            onChangeFacingMode={setFacingMode}
            excludedClasses={excludedClasses}
            basketItems={basketItems}
            onScanUpdate={handleScanUpdate}
            onDeleteItem={handleDeleteItem}
            onResetSession={handleResetSession}
            onGoToBasket={function openBasketTab() { navigateToTab('basket') }}
            onGenerateRecipes={handleGenerateRecipes}
            onRegisterWsSend={registerWsSend}
            onLogEvent={appendLog}
          />
        )}

        {activeTab === 'basket' && (
          <BasketView
            items={basketItems}
            totalCalories={basketTotalCalories}
            excludedClasses={excludedClasses}
            isGeneratingRecipes={recipeState.status === 'loading'}
            onIncrement={handleIncrementItem}
            onDecrement={handleDecrementItem}
            onDelete={handleDeleteItem}
            onRestore={handleRestoreItem}
            onAddManual={handleAddManualItem}
            onClearAll={handleClearBasket}
            onGenerateRecipes={handleGenerateRecipes}
            onGoToScanner={function openScannerTab() { navigateToTab('scanner') }}
          />
        )}

        {activeTab === 'recipes' && (
          <RecipesView
            recipeState={recipeState}
            checkedSteps={checkedSteps}
            onToggleStep={handleToggleRecipeStep}
            onRemoveRecipe={handleRemoveRecipe}
            onRetry={handleGenerateRecipes}
            onGoToBasket={function openBasketFromRecipes() { navigateToTab('basket') }}
            onGoToScanner={function openScannerFromRecipes() { navigateToTab('scanner') }}
          />
        )}
      </main>

      <nav className="mobile-bottom-nav" aria-label="Mobile navigation">
        <button
          type="button"
          className={`mobile-nav-btn ${activeTab === 'scanner' ? 'mobile-nav-btn-active' : ''}`}
          onClick={function clickMobileScanner() { navigateToTab('scanner') }}
        >
          <ScanLine size={18} />
          <span>Scanner</span>
        </button>

        <button
          type="button"
          className={`mobile-nav-btn ${activeTab === 'basket' ? 'mobile-nav-btn-active' : ''}`}
          onClick={function clickMobileBasket() { navigateToTab('basket') }}
        >
          <ShoppingBasket size={18} />
          <span>Basket</span>
          <span className="mobile-nav-badge">{basketItems.length}</span>
        </button>

        <button
          type="button"
          className={`mobile-nav-btn ${activeTab === 'recipes' ? 'mobile-nav-btn-active' : ''}`}
          onClick={function clickMobileRecipes() { navigateToTab('recipes') }}
        >
          <ChefHat size={18} />
          <span>Recipes</span>
          {recipeCount > 0 && <span className="mobile-nav-badge">{recipeCount}</span>}
        </button>
      </nav>

      <DiagnosticsModal
        isOpen={diagnosticsOpen}
        healthState={healthState}
        logs={logs}
        onClose={handleCloseDiagnostics}
        onRefreshHealth={refreshHealth}
        onClearLogs={function clearAllLogs() { setLogs([]) }}
      />
    </div>
  )
}
