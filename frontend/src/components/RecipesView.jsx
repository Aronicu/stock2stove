import {
  CheckSquare,
  ChefHat,
  Clock,
  Flame,
  Gauge,
  Lightbulb,
  RefreshCw,
  ScanLine,
  Square,
  Warehouse,
} from 'lucide-react'

export default function RecipesView({
  recipeState,
  checkedSteps,
  onToggleStep,
  onRetry,
  onGoToScanner,
}) {
  if (recipeState.status === 'idle') {
    return (
      <div className="card empty-state-card">
        <ChefHat size={48} className="empty-icon" />
        <h3>No Recipes Generated Yet</h3>
        <p>
          Scan ingredients or curate your basket, then click &ldquo;Generate Recipes&rdquo; for zero-waste meal plans and calorie breakdowns.
        </p>
        <button type="button" className="btn btn-primary" onClick={onGoToScanner}>
          <ScanLine size={16} />
          <span>Go to Scanner</span>
        </button>
      </div>
    )
  }

  if (recipeState.status === 'loading') {
    return (
      <div className="card empty-state-card">
        <div className="spinner" aria-hidden="true" />
        <h3>Crafting Zero-Waste Recipes...</h3>
        <p>Sending your post-deletion ingredient basket and calorie budget to the chef engine.</p>
      </div>
    )
  }

  if (recipeState.status === 'error') {
    return (
      <div className="card error-state-card">
        <h3>Recipe Generation Failed</h3>
        <p>{recipeState.error || 'Could not reach the recipe generation endpoint.'}</p>
        <div className="action-row">
          <button type="button" className="btn btn-primary" onClick={onRetry}>
            <RefreshCw size={16} />
            <span>Retry</span>
          </button>
        </div>
      </div>
    )
  }

  const data = recipeState.data || {}
  const recipes = Array.isArray(data.recipes) ? data.recipes : []
  const inventoryUsed = Array.isArray(data.inventory_used) ? data.inventory_used : []
  const totalInvCals = Number(data.total_inventory_calories || 0)

  return (
    <div className="recipes-layout">
      <div className="card telemetry-banner">
        <div className="telemetry-left">
          <span className="telemetry-pill telemetry-source">
            LLM: {data.source || 'offline_cache'}
          </span>
          <span className={`telemetry-pill ${data.cached ? 'telemetry-cached' : 'telemetry-live'}`}>
            {data.cached ? 'Cached' : 'Live'}
          </span>
          <span className="telemetry-pill telemetry-latency">
            {Math.round(Number(data.latency_ms || 0))} ms
          </span>
          <span className="telemetry-pill telemetry-calories">
            <Flame size={14} />
            <span>Inventory: {totalInvCals} kcal</span>
          </span>
        </div>

        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onRetry}
        >
          <RefreshCw size={15} />
          <span>Regenerate</span>
        </button>

        {inventoryUsed.length > 0 && (
          <div className="inventory-used-row">
            <span className="used-label">Ingredients sent:</span>
            {inventoryUsed.map(function renderUsedChip(clsId) {
              return (
                <span key={clsId} className="used-chip">
                  {clsId}
                </span>
              )
            })}
          </div>
        )}
      </div>

      <div className="recipes-grid">
        {recipes.map(function renderRecipeCard(recipe, cardIdx) {
          const recipeId = recipe.id || `recipe-${cardIdx}`
          const steps = Array.isArray(recipe.steps) ? recipe.steps : []
          const checkedSet = checkedSteps[recipeId] || []
          const completedCount = checkedSet.length
          const progressPct = steps.length > 0 ? Math.round((completedCount / steps.length) * 100) : 0
          const ingredientCalories = recipe.ingredient_calories && typeof recipe.ingredient_calories === 'object'
            ? Object.entries(recipe.ingredient_calories)
            : []

          return (
            <article key={recipeId} className="card recipe-card">
              <div className="recipe-header">
                <h3>{recipe.title}</h3>
                {recipe.estimated_calories != null && (
                  <span className="recipe-kcal-badge">
                    <Flame size={15} />
                    <span>{recipe.estimated_calories} kcal</span>
                  </span>
                )}
              </div>

              <div className="recipe-meta-chips">
                <span className="meta-chip meta-chip-primary">
                  <Clock size={14} />
                  <span>{recipe.prep_time_mins} mins</span>
                </span>
                <span className="meta-chip meta-chip-secondary">
                  <Gauge size={14} />
                  <span>{recipe.difficulty}</span>
                </span>
                {(recipe.storage_zones_used || []).map(function renderZone(zone) {
                  return (
                    <span key={zone} className="meta-chip meta-chip-zone">
                      <Warehouse size={14} />
                      <span>{zone}</span>
                    </span>
                  )
                })}
              </div>

              {ingredientCalories.length > 0 && (
                <div className="recipe-section">
                  <span className="section-label">Calorie Breakdown</span>
                  <div className="pill-wrap">
                    {ingredientCalories.map(function renderIngCal([ingName, kcalVal]) {
                      return (
                        <span key={ingName} className="kcal-pill">
                          {ingName}: <strong>{kcalVal} kcal</strong>
                        </span>
                      )
                    })}
                  </div>
                </div>
              )}

              {recipe.matched_ingredients && recipe.matched_ingredients.length > 0 && (
                <div className="recipe-section">
                  <span className="section-label">Matched From Your Stock</span>
                  <div className="pill-wrap">
                    {recipe.matched_ingredients.map(function renderMatched(ing) {
                      return (
                        <span key={ing} className="matched-pill">
                          {ing}
                        </span>
                      )
                    })}
                  </div>
                </div>
              )}

              {recipe.additional_staples && recipe.additional_staples.length > 0 && (
                <div className="recipe-section">
                  <span className="section-label">Pantry Staples</span>
                  <div className="pill-wrap">
                    {recipe.additional_staples.map(function renderStaple(staple) {
                      return (
                        <span key={staple} className="staple-pill">
                          {staple}
                        </span>
                      )
                    })}
                  </div>
                </div>
              )}

              {steps.length > 0 && (
                <div className="recipe-section">
                  <div className="steps-header">
                    <span className="section-label">Cooking Steps</span>
                    <span className="steps-progress-text">
                      {completedCount}/{steps.length} Done
                    </span>
                  </div>
                  <div className="progress-track" aria-label="Cooking progress">
                    <div
                      className="progress-fill"
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                  <ul className="steps-list">
                    {steps.map(function renderStep(stepText, stepIdx) {
                      const isDone = checkedSet.includes(stepIdx)
                      return (
                        <li key={stepIdx}>
                          <button
                            type="button"
                            className={`step-row-btn ${isDone ? 'step-row-done' : ''}`}
                            onClick={function clickStep() { onToggleStep(recipeId, stepIdx) }}
                          >
                            {isDone ? <CheckSquare size={18} className="step-check-on" /> : <Square size={18} />}
                            <span>{stepIdx + 1}. {stepText}</span>
                          </button>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}

              {recipe.zero_waste_tip && (
                <div className="zero-waste-callout">
                  <div className="tip-icon-badge">
                    <Lightbulb size={16} />
                  </div>
                  <div>
                    <strong>Zero-Waste Chef Tip</strong>
                    <p>{recipe.zero_waste_tip}</p>
                  </div>
                </div>
              )}
            </article>
          )
        })}
      </div>
    </div>
  )
}
