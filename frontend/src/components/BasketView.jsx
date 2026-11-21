import { useState } from 'react'
import {
  ArrowLeft,
  Flame,
  Minus,
  Package,
  Plus,
  RotateCcw,
  ScanLine,
  ShoppingBasket,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react'
import {
  UNIFIED_FOOD_CLASSES,
  formatCropDataUrl,
  getCaloriesPerUnit,
  getClassMeta,
} from '../constants/catalog.js'

export default function BasketView({
  items,
  totalCalories,
  excludedClasses,
  isGeneratingRecipes,
  onIncrement,
  onDecrement,
  onDelete,
  onRestore,
  onAddManual,
  onClearAll,
  onGenerateRecipes,
  onGoToScanner,
}) {
  const [showAddModal, setShowAddModal] = useState(false)
  const [showClearModal, setShowClearModal] = useState(false)
  const [selectedClassId, setSelectedClassId] = useState(UNIFIED_FOOD_CLASSES[0].class_id)
  const [manualCount, setManualCount] = useState(1)

  let totalUnits = 0
  for (let i = 0; i < items.length; i += 1) {
    totalUnits += Math.max(1, Number(items[i].count || 1))
  }

  function handleConfirmAdd() {
    onAddManual(selectedClassId, manualCount)
    setManualCount(1)
    setShowAddModal(false)
  }

  function handleConfirmClear() {
    setShowClearModal(false)
    onClearAll()
  }

  return (
    <div className="basket-layout">
      <div className="nav-back-row">
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onGoToScanner}
        >
          <ArrowLeft size={15} />
          <span>Back to Scanner</span>
        </button>
      </div>
      <div className="calorie-summary-banner card">
        <div className="summary-stat-group">
          <div className="summary-stat">
            <span className="summary-label">Categories</span>
            <strong className="summary-val">{items.length}</strong>
          </div>
          <div className="summary-stat">
            <span className="summary-label">Total Units</span>
            <strong className="summary-val">{totalUnits}</strong>
          </div>
          <div className="summary-stat summary-stat-highlight">
            <span className="summary-label">
              <Flame size={15} />
              <span>Total Estimated Calories</span>
            </span>
            <strong className="summary-val-kcal">{totalCalories} kcal</strong>
          </div>
        </div>

        <div className="summary-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={function openAdd() { setShowAddModal(true) }}
          >
            <Plus size={16} />
            <span>Add Item</span>
          </button>

          {items.length > 0 && (
            <button
              type="button"
              className="btn btn-ghost-danger"
              onClick={function openClear() { setShowClearModal(true) }}
            >
              <Trash2 size={16} />
              <span>Clear All</span>
            </button>
          )}

          <button
            type="button"
            className="btn btn-primary"
            disabled={items.length === 0 || isGeneratingRecipes}
            onClick={onGenerateRecipes}
          >
            <Sparkles size={16} />
            <span>{isGeneratingRecipes ? 'Generating...' : 'Generate Recipes'}</span>
          </button>
        </div>
      </div>

      {excludedClasses && excludedClasses.length > 0 && (
        <div className="excluded-bar card">
          <span className="excluded-title">Removed from session (hidden from live re-detection):</span>
          <div className="excluded-pills">
            {excludedClasses.map(function renderExcluded(clsId) {
              const meta = getClassMeta(clsId)
              return (
                <button
                  key={clsId}
                  type="button"
                  className="restore-pill"
                  onClick={function clickRestore() { onRestore(clsId) }}
                  title={`Restore ${meta.display_name}`}
                >
                  <span>{meta.display_name}</span>
                  <RotateCcw size={13} />
                  <span>Restore</span>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {items.length === 0 ? (
        <div className="card empty-state-card">
          <ShoppingBasket size={46} className="empty-icon" />
          <h3>Your Ingredient Basket is Empty</h3>
          <p>
            Scan pantry or refrigerator items in the Scanner tab, or tap &ldquo;+ Add Item&rdquo; to add ingredients manually.
          </p>
          <div className="action-row">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onGoToScanner}
            >
              <ScanLine size={16} />
              <span>Go to Scanner</span>
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={function openAddEmpty() { setShowAddModal(true) }}
            >
              <Plus size={16} />
              <span>Add Item</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="basket-grid">
          {items.map(function renderBasketCard(item) {
            const meta = getClassMeta(item.class_id)
            const displayName = item.display_name || meta.display_name
            const zone = item.storage_zone || meta.storage_zone
            const cropSrc = formatCropDataUrl(item.crop_jpeg_b64)
            const qty = Math.max(1, Number(item.count || 1))
            const perUnit = Number(item.calories_per_unit || getCaloriesPerUnit(item.class_id))
            const subtotal = item.estimated_calories != null ? Number(item.estimated_calories) : perUnit * qty
            const confPct = Math.round(Number(item.max_conf || 0) * 100)
            const itemKey = item.item_key || item.class_id

            return (
              <div key={itemKey} className="card basket-item-card">
                <div className="basket-item-top">
                  {cropSrc ? (
                    <img src={cropSrc} alt={displayName} className="basket-crop-img" />
                  ) : (
                    <div className="basket-crop-placeholder">
                      <Package size={24} />
                    </div>
                  )}

                  <div className="basket-item-info">
                    <div className="basket-item-title-row">
                      <h3>{displayName}</h3>
                      <span className={`zone-badge ${zone === 'Refrigerator' ? 'zone-fridge' : 'zone-pantry'}`}>
                        {zone}
                      </span>
                    </div>
                    <div className="basket-meta-row">
                      {confPct > 0 && <span className="conf-tag">{confPct}% confidence</span>}
                      <span className="kcal-unit-tag">{perUnit} kcal / unit</span>
                    </div>
                  </div>

                  <button
                    type="button"
                    className="btn-icon-danger"
                    onClick={function clickRemoveItem() { onDelete(itemKey, false) }}
                    title={`Delete ${displayName}`}
                    aria-label={`Delete ${displayName}`}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>

                <div className="basket-item-bottom">
                  <div className="stepper-group" aria-label={`Quantity for ${displayName}`}>
                    <button
                      type="button"
                      className="stepper-btn"
                      onClick={function clickDec() { onDecrement(itemKey) }}
                      aria-label={`Decrease ${displayName}`}
                    >
                      <Minus size={15} />
                    </button>
                    <span className="stepper-val">{qty}</span>
                    <button
                      type="button"
                      className="stepper-btn"
                      onClick={function clickInc() { onIncrement(itemKey) }}
                      aria-label={`Increase ${displayName}`}
                    >
                      <Plus size={15} />
                    </button>
                  </div>

                  <div className="subtotal-pill">
                    <Flame size={15} />
                    <strong>{subtotal} kcal</strong>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {showAddModal && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="add-item-title">
          <div className="modal-card">
            <div className="modal-header">
              <h3 id="add-item-title">Add Food Item</h3>
              <button
                type="button"
                className="btn-close"
                onClick={function closeAdd() { setShowAddModal(false) }}
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </div>

            <div className="modal-body">
              <p className="helper-text">Select one of the 8 unified kitchen ingredient classes:</p>
              <div className="class-picker-list">
                {UNIFIED_FOOD_CLASSES.map(function renderClassOption(cls) {
                  const isSelected = selectedClassId === cls.class_id
                  return (
                    <button
                      key={cls.class_id}
                      type="button"
                      className={`class-picker-row ${isSelected ? 'class-picker-row-active' : ''}`}
                      onClick={function pickClass() { setSelectedClassId(cls.class_id) }}
                    >
                      <div>
                        <strong>{cls.display_name}</strong>
                        <small>{cls.storage_zone}</small>
                      </div>
                      <span className="kcal-unit-tag">{cls.calories_per_unit} kcal</span>
                    </button>
                  )
                })}
              </div>

              <div className="modal-qty-row">
                <span>Quantity:</span>
                <div className="stepper-group">
                  <button
                    type="button"
                    className="stepper-btn"
                    onClick={function decManual() { setManualCount(function clamp(c) { return Math.max(1, c - 1) }) }}
                  >
                    <Minus size={15} />
                  </button>
                  <span className="stepper-val">{manualCount}</span>
                  <button
                    type="button"
                    className="stepper-btn"
                    onClick={function incManual() { setManualCount(function add(c) { return c + 1 }) }}
                  >
                    <Plus size={15} />
                  </button>
                </div>
              </div>
            </div>

            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={function cancelAdd() { setShowAddModal(false) }}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleConfirmAdd}
              >
                Add to Basket
              </button>
            </div>
          </div>
        </div>
      )}

      {showClearModal && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="clear-basket-title">
          <div className="modal-card modal-card-sm">
            <div className="modal-header">
              <h3 id="clear-basket-title">Clear All Items?</h3>
              <button
                type="button"
                className="btn-close"
                onClick={function closeClear() { setShowClearModal(false) }}
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p>This will remove all {items.length} ingredient categories from your basket and reset session exclusions.</p>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={function keepItems() { setShowClearModal(false) }}
              >
                Keep Items
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={handleConfirmClear}
              >
                Clear All
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
