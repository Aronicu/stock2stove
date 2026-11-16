package com.kaggleing.stock2stove.ui.recipes

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.kaggleing.stock2stove.data.api.InferenceApiService
import com.kaggleing.stock2stove.data.api.NetworkClientProvider
import com.kaggleing.stock2stove.data.model.InventoryItemRequest
import com.kaggleing.stock2stove.data.model.RecipeGenerateRequest
import com.kaggleing.stock2stove.data.model.RecipeGenerateResponse
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface RecipeUiState {
    data object Idle : RecipeUiState
    data object Loading : RecipeUiState
    data class Success(
        val response: RecipeGenerateResponse,
        val checkedSteps: Map<String, Set<Int>> = emptyMap()
    ) : RecipeUiState
    data class Error(
        val message: String,
        val canRetry: Boolean = true
    ) : RecipeUiState
}

class RecipeViewModel(
    private val apiService: InferenceApiService = NetworkClientProvider.apiService
) : ViewModel() {
    private val _uiState = MutableStateFlow<RecipeUiState>(RecipeUiState.Idle)
    val uiState: StateFlow<RecipeUiState> = _uiState.asStateFlow()

    private var lastItems: List<InventoryItemRequest> = emptyList()
    private var lastTimeoutSec: Float = 2.5f

    fun generateRecipes(items: List<InventoryItemRequest>, timeoutSec: Float = 2.5f) {
        lastItems = items
        lastTimeoutSec = timeoutSec
        _uiState.value = RecipeUiState.Loading

        viewModelScope.launch {
            try {
                val request = RecipeGenerateRequest(
                    confirmed_inventory = items,
                    timeout_sec = timeoutSec
                )
                val response = apiService.generateRecipes(request)
                com.kaggleing.stock2stove.data.repository.AlphaTelemetryLogger.getDefault().logEvent(
                    "RECIPE",
                    "INFO",
                    "Generated " + response.recipes.size + " recipes via " + response.source
                )
                _uiState.value = RecipeUiState.Success(response = response)
            } catch (e: Exception) {
                _uiState.value = RecipeUiState.Error(
                    message = e.localizedMessage ?: "Failed to generate recipes"
                )
            }
        }
    }

    fun retry() {
        if (lastItems.isNotEmpty()) {
            generateRecipes(lastItems, lastTimeoutSec)
        }
    }

    fun toggleStep(recipeId: String, stepIndex: Int) {
        val current = _uiState.value
        if (current is RecipeUiState.Success) {
            val currentSteps = current.checkedSteps[recipeId]?.toMutableSet() ?: mutableSetOf()
            if (currentSteps.contains(stepIndex)) {
                currentSteps.remove(stepIndex)
            } else {
                currentSteps.add(stepIndex)
            }
            val updatedMap = current.checkedSteps.toMutableMap()
            updatedMap[recipeId] = currentSteps
            _uiState.value = current.copy(checkedSteps = updatedMap)
        }
    }

    fun isStepChecked(recipeId: String, stepIndex: Int): Boolean {
        val current = _uiState.value
        if (current is RecipeUiState.Success) {
            return current.checkedSteps[recipeId]?.contains(stepIndex) == true
        }
        return false
    }

    fun injectRecipes(response: RecipeGenerateResponse) {
        _uiState.value = RecipeUiState.Success(response = response)
    }

    fun resetState() {
        _uiState.value = RecipeUiState.Idle
    }
}
