import {
  Activity,
  ChefHat,
  ScanLine,
  ShoppingBasket,
} from 'lucide-react'

export default function HeaderBar({
  activeTab,
  onSelectTab,
  basketCount,
  basketTotalCalories,
  recipeCount,
  healthState,
  onOpenDiagnostics,
}) {
  const status = healthState.status
  let statusDotClass = 'status-dot-idle'
  let statusText = 'Checking...'
  if (status === 'connected') {
    statusDotClass = 'status-dot-ok'
    statusText = healthState.latencyMs != null ? `Server Online · ${healthState.latencyMs} ms` : 'Server Online'
  } else if (status === 'error') {
    statusDotClass = 'status-dot-err'
    statusText = 'Server Unreachable'
  }

  const activeWeights = healthState.data?.model_status?.active_weights || 'YOLOv8n / Gemini'

  return (
    <header className="app-header">
      <div className="app-header-inner">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <ChefHat size={22} />
          </div>
          <div>
            <h1 className="brand-title">Stock-to-Stove</h1>
            <p className="brand-subtitle">Zero-Waste Pantry Vision &amp; Calorie-Aware Chef</p>
          </div>
        </div>

        <nav className="primary-tabs" aria-label="Primary sections">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'scanner' ? 'tab-btn-active' : ''}`}
            onClick={function handleScannerTab() { onSelectTab('scanner') }}
          >
            <ScanLine size={17} />
            <span>Scanner</span>
          </button>

          <button
            type="button"
            className={`tab-btn ${activeTab === 'basket' ? 'tab-btn-active' : ''}`}
            onClick={function handleBasketTab() { onSelectTab('basket') }}
          >
            <ShoppingBasket size={17} />
            <span>Basket</span>
            <span className="tab-pill">
              {basketCount} · {basketTotalCalories} kcal
            </span>
          </button>

          <button
            type="button"
            className={`tab-btn ${activeTab === 'recipes' ? 'tab-btn-active' : ''}`}
            onClick={function handleRecipesTab() { onSelectTab('recipes') }}
          >
            <ChefHat size={17} />
            <span>Recipes</span>
            {recipeCount > 0 && <span className="tab-pill">{recipeCount}</span>}
          </button>
        </nav>

        <button
          type="button"
          className="health-pill"
          onClick={onOpenDiagnostics}
          title="Open Server Health & Diagnostics"
        >
          <span className={`status-dot ${statusDotClass}`} />
          <span className="health-pill-text">
            <strong>{statusText}</strong>
            <small>{activeWeights}</small>
          </span>
          <Activity size={16} className="health-pill-icon" />
        </button>
      </div>
    </header>
  )
}
