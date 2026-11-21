import { useCallback, useEffect, useRef, useState } from 'react'
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
  computeTotalBasketCalories,
  createClientSessionId,
  getCaloriesPerUnit,
  getClassMeta,
  normalizeClassId,
} from './constants/catalog.js'

function mergeBasketInventory(prevItems, incomingItems, excludedList) {
  const excludedSet = new Set(excludedList.map(normalizeClassId))
  const byClass = new Map()

  for (let i = 0; i < prevItems.length; i += 1) {
    const prev = prevItems[i]
    const normId = normalizeClassId(prev.class_id)
    if (!normId || excludedSet.has(normId)) {
      continue
    }
    byClass.set(normId, prev)
  }

  for (let i = 0; i < incomingItems.length; i += 1) {
    const inc = incomingItems[i]
    const normId = normalizeClassId(inc.class_id)
    if (!normId || excludedSet.has(normId)) {
      continue
    }
    const meta = getClassMeta(normId)
    const perUnit = Number(inc.calories_per_unit || meta.calories_per_unit || getCaloriesPerUnit(normId))
    const incCount = Math.max(1, Number(inc.count || 1))
    const incConf = Number(inc.max_conf || 0)
    const existing = byClass.get(normId)

    if (existing) {
      const nextCount = Math.max(Number(existing.count || 1), incCount)
      const nextConf = Math.max(Number(existing.max_conf || 0), incConf)
      const nextCrop = inc.crop_jpeg_b64 && (incConf >= Number(existing.max_conf || 0) || !existing.crop_jpeg_b64)
        ? inc.crop_jpeg_b64
        : existing.crop_jpeg_b64
      byClass.set(normId, {
        class_id: normId,
        display_name: inc.display_name || existing.display_name || meta.display_name,
        storage_zone: existing.storage_zone || meta.storage_zone,
        count: nextCount,
        max_conf: nextConf,
        crop_jpeg_b64: nextCrop || null,
        calories_per_unit: perUnit,
        estimated_calories: perUnit * nextCount,
      })
    } else {
      byClass.set(normId, {
        class_id: normId,
        display_name: inc.display_name || meta.display_name,
        storage_zone: meta.storage_zone,
        count: incCount,
        max_conf: incConf,
        crop_jpeg_b64: inc.crop_jpeg_b64 || null,
        calories_per_unit: perUnit,
        estimated_calories: perUnit * incCount,
      })
    }
  }

  return Array.from(byClass.values())
}

export default function App() {
  const [activeTab, setActiveTab] = useState('scanner')
  const [sessionId] = useState(createClientSessionId)
  const [detectorMode, setDetectorMode] = useState('yolov8')
  const [enableClahe, setEnableClahe] = useState(true)
  const [confThreshold, setConfThreshold] = useState(0.45)

  const [basketItems, setBasketItems] = useState([])
  const [excludedClasses, setExcludedClasses] = useState([])
  const excludedRef = useRef([])
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

  function registerWsSend(fn) {
    wsSendRef.current = fn
  }

  function handleScanUpdate({ confirmed_inventory, excluded_classes, recipes, replaceInventory }) {
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

    if (replaceInventory) {
      const excSet = new Set(combinedExcluded)
      setBasketItems(function filterOnly(prev) {
        return prev.filter(function keepRow(row) {
          return !excSet.has(normalizeClassId(row.class_id))
        })
      })
      return
    }

    if (Array.isArray(confirmed_inventory) && confirmed_inventory.length > 0) {
      setBasketItems(function mergeNext(prev) {
        return mergeBasketInventory(prev, confirmed_inventory, combinedExcluded)
      })
    }

    if (recipes && Array.isArray(recipes.recipes) && recipes.recipes.length > 0) {
      const usedFromConfirm = Array.isArray(confirmed_inventory)
        ? confirmed_inventory
            .map(function mapInvId(it) { return normalizeClassId(it.class_id) })
            .filter(Boolean)
        : []
      setRecipeState({
        status: 'success',
        data: {
          source: recipes.recipe_source || recipes.source || 'offline_cache',
          cached: true,
          latency_ms: recipes.inference_ms || 0,
          inventory_used: usedFromConfirm,
          total_inventory_calories: recipes.total_inventory_calories || recipes.total_estimated_calories || 0,
          recipes: recipes.recipes,
        },
        error: '',
      })
    }
  }

  async function handleDeleteItem(rawClassId) {
    const normId = normalizeClassId(rawClassId)
    if (!normId) {
      return
    }
    const nextExcluded = Array.from(new Set(excludedRef.current.concat([normId])))
    excludedRef.current = nextExcluded
    setExcludedClasses(nextExcluded)

    setBasketItems(function removeBasketRow(prev) {
      return prev.filter(function keepOther(item) {
        return normalizeClassId(item.class_id) !== normId
      })
    })
    appendLog('INFO', 'BASKET', `Removed ${normId} from basket and excluded from session`)

    if (wsSendRef.current) {
      wsSendRef.current({
        action: 'delete_item',
        class_id: normId,
      })
    }

    try {
      await deleteSessionItemRest(sessionId, normId)
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

  function handleIncrementItem(classId) {
    const normId = normalizeClassId(classId)
    setBasketItems(function incRow(prev) {
      return prev.map(function updateRow(item) {
        if (normalizeClassId(item.class_id) !== normId) {
          return item
        }
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
  }

  function handleDecrementItem(classId) {
    const normId = normalizeClassId(classId)
    const target = basketItems.find(function findRow(item) {
      return normalizeClassId(item.class_id) === normId
    })
    if (target && Number(target.count || 1) <= 1) {
      handleDeleteItem(normId)
      return
    }
    setBasketItems(function decRow(prev) {
      return prev.map(function updateRow(item) {
        if (normalizeClassId(item.class_id) !== normId) {
          return item
        }
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
  }

  function handleAddManualItem(classId, count) {
    const normId = normalizeClassId(classId)
    const qty = Math.max(1, Number(count || 1))
    const meta = getClassMeta(normId)
    const perUnit = meta.calories_per_unit || getCaloriesPerUnit(normId)

    if (excludedRef.current.includes(normId)) {
      handleRestoreItem(normId)
    }

    setBasketItems(function upsertManual(prev) {
      let found = false
      const updated = prev.map(function mapRow(item) {
        if (normalizeClassId(item.class_id) !== normId) {
          return item
        }
        found = true
        const nextCount = Number(item.count || 1) + qty
        return {
          ...item,
          count: nextCount,
          calories_per_unit: perUnit,
          estimated_calories: perUnit * nextCount,
        }
      })
      if (!found) {
        updated.push({
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
    appendLog('INFO', 'BASKET', `Manually added ${meta.display_name} x${qty}`)
  }

  async function handleResetSession() {
    excludedRef.current = []
    setExcludedClasses([])
    setBasketItems([])
    appendLog('INFO', 'SESSION', `Reset session ${sessionId}`)
    try {
      await resetBackendSession(sessionId)
    } catch (err) {
      appendLog('WARN', 'SESSION', `Reset sync failed: ${err.message}`)
    }
  }

  async function handleClearBasket() {
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
    setActiveTab('recipes')
    setRecipeState({ status: 'loading', data: null, error: '' })
    try {
      const response = await generateRecipesFromBasket(basketItems, 8.0)
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

  function handleOpenDiagnostics() {
    setDiagnosticsOpen(true)
    refreshHealth()
  }

  const basketTotalCalories = computeTotalBasketCalories(basketItems)
  const recipeCount = recipeState.data && Array.isArray(recipeState.data.recipes)
    ? recipeState.data.recipes.length
    : 0

  return (
    <div className="app-shell">
      <HeaderBar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
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
            excludedClasses={excludedClasses}
            basketItems={basketItems}
            onScanUpdate={handleScanUpdate}
            onDeleteItem={handleDeleteItem}
            onResetSession={handleResetSession}
            onGoToBasket={function openBasketTab() { setActiveTab('basket') }}
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
            onGoToScanner={function openScannerTab() { setActiveTab('scanner') }}
          />
        )}

        {activeTab === 'recipes' && (
          <RecipesView
            recipeState={recipeState}
            checkedSteps={checkedSteps}
            onToggleStep={handleToggleRecipeStep}
            onRetry={handleGenerateRecipes}
            onGoToScanner={function openScannerFromRecipes() { setActiveTab('scanner') }}
          />
        )}
      </main>

      <DiagnosticsModal
        isOpen={diagnosticsOpen}
        healthState={healthState}
        logs={logs}
        onClose={function closeDiag() { setDiagnosticsOpen(false) }}
        onRefreshHealth={refreshHealth}
        onClearLogs={function clearAllLogs() { setLogs([]) }}
      />
    </div>
  )
}
