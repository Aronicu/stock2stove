package com.kaggleing.stock2stove

import com.kaggleing.stock2stove.data.api.InferenceApiService
import com.kaggleing.stock2stove.data.api.NetworkClientProvider
import com.kaggleing.stock2stove.data.model.ConfirmedItem
import com.kaggleing.stock2stove.data.model.HealthResponse
import com.kaggleing.stock2stove.data.model.InventoryItemRequest
import com.kaggleing.stock2stove.data.model.RecipeCard
import com.kaggleing.stock2stove.data.model.RecipeGenerateRequest
import com.kaggleing.stock2stove.data.model.RecipeGenerateResponse
import com.kaggleing.stock2stove.data.model.StorageZone
import com.kaggleing.stock2stove.data.model.UnifiedItemClass
import com.kaggleing.stock2stove.ui.inventory.InventoryViewModel
import com.kaggleing.stock2stove.ui.recipes.RecipeUiState
import com.kaggleing.stock2stove.ui.recipes.RecipeViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import kotlinx.serialization.encodeToString
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.IOException

@OptIn(ExperimentalCoroutinesApi::class)
class InventoryRecipeTest {
    private val testDispatcher = StandardTestDispatcher()

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun testUnifiedItemClassEntriesAndZones() {
        assertEquals(8, UnifiedItemClass.entries.size)

        assertEquals("pasta_noodles", UnifiedItemClass.PASTA_NOODLES.classId)
        assertEquals(StorageZone.PANTRY, UnifiedItemClass.PASTA_NOODLES.defaultStorageZone)

        assertEquals("canned_tomato", UnifiedItemClass.CANNED_TOMATO.classId)
        assertEquals(StorageZone.PANTRY, UnifiedItemClass.CANNED_TOMATO.defaultStorageZone)

        assertEquals("canned_tuna_meat", UnifiedItemClass.CANNED_TUNA_MEAT.classId)
        assertEquals(StorageZone.PANTRY, UnifiedItemClass.CANNED_TUNA_MEAT.defaultStorageZone)

        assertEquals("cereal_grains", UnifiedItemClass.CEREAL_GRAINS.classId)
        assertEquals(StorageZone.PANTRY, UnifiedItemClass.CEREAL_GRAINS.defaultStorageZone)

        assertEquals("coffee_condiment", UnifiedItemClass.COFFEE_CONDIMENT.classId)
        assertEquals(StorageZone.PANTRY, UnifiedItemClass.COFFEE_CONDIMENT.defaultStorageZone)

        assertEquals("milk_dairy", UnifiedItemClass.MILK_DAIRY.classId)
        assertEquals(StorageZone.REFRIGERATOR, UnifiedItemClass.MILK_DAIRY.defaultStorageZone)

        assertEquals("eggs", UnifiedItemClass.EGGS.classId)
        assertEquals(StorageZone.REFRIGERATOR, UnifiedItemClass.EGGS.defaultStorageZone)

        assertEquals("fresh_produce", UnifiedItemClass.FRESH_PRODUCE.classId)
        assertEquals(StorageZone.REFRIGERATOR, UnifiedItemClass.FRESH_PRODUCE.defaultStorageZone)
    }

    @Test
    fun testUnifiedItemClassNormalization() {
        assertEquals("pasta_noodles", UnifiedItemClass.normalizeClassId("pasta_box"))
        assertEquals("pasta_noodles", UnifiedItemClass.normalizeClassId("SPAGHETTI"))
        assertEquals("pasta_noodles", UnifiedItemClass.normalizeClassId("instant-noodles"))

        assertEquals("canned_tomato", UnifiedItemClass.normalizeClassId("tomato_can"))
        assertEquals("canned_tomato", UnifiedItemClass.normalizeClassId("canned tomatoes"))
        assertEquals("canned_tomato", UnifiedItemClass.normalizeClassId("tomato_sauce"))

        assertEquals("canned_tuna_meat", UnifiedItemClass.normalizeClassId("tuna_can"))
        assertEquals("canned_tuna_meat", UnifiedItemClass.normalizeClassId("sardines"))
        assertEquals("canned_tuna_meat", UnifiedItemClass.normalizeClassId("corned_beef"))

        assertEquals("cereal_grains", UnifiedItemClass.normalizeClassId("cereal_box"))
        assertEquals("cereal_grains", UnifiedItemClass.normalizeClassId("oats"))
        assertEquals("cereal_grains", UnifiedItemClass.normalizeClassId("rice_pack"))

        assertEquals("coffee_condiment", UnifiedItemClass.normalizeClassId("coffee_bag"))
        assertEquals("coffee_condiment", UnifiedItemClass.normalizeClassId("cooking_oil"))
        assertEquals("coffee_condiment", UnifiedItemClass.normalizeClassId("soy_sauce"))

        assertEquals("milk_dairy", UnifiedItemClass.normalizeClassId("milk_carton"))
        assertEquals("milk_dairy", UnifiedItemClass.normalizeClassId("cheese_block"))
        assertEquals("milk_dairy", UnifiedItemClass.normalizeClassId("butter"))

        assertEquals("eggs", UnifiedItemClass.normalizeClassId("egg"))
        assertEquals("eggs", UnifiedItemClass.normalizeClassId("egg_carton"))

        assertEquals("fresh_produce", UnifiedItemClass.normalizeClassId("garlic"))
        assertEquals("fresh_produce", UnifiedItemClass.normalizeClassId("onion"))
        assertEquals("fresh_produce", UnifiedItemClass.normalizeClassId("vegetables"))

        assertEquals("pasta_noodles", UnifiedItemClass.normalizeClassId(null))
        assertEquals("pasta_noodles", UnifiedItemClass.normalizeClassId(""))
        assertEquals("pasta_noodles", UnifiedItemClass.normalizeClassId("   "))
    }

    @Test
    fun testUnifiedItemClassFromClassId() {
        val entry = UnifiedItemClass.fromClassId("eggs")
        assertNotNull(entry)
        assertEquals(UnifiedItemClass.EGGS, entry)

        val aliasEntry = UnifiedItemClass.fromClassId("tomato_can")
        assertNotNull(aliasEntry)
        assertEquals(UnifiedItemClass.CANNED_TOMATO, aliasEntry)

        val nullEntry = UnifiedItemClass.fromClassId(null)
        assertNull(nullEntry)

        val blankEntry = UnifiedItemClass.fromClassId("  ")
        assertNull(blankEntry)
    }

    @Test
    fun testInventoryViewModelInitialInventoryAndMerging() {
        val viewModel = InventoryViewModel()
        val rawItems = listOf(
            ConfirmedItem(class_id = "pasta_noodles", display_name = "Pasta & Noodles", count = 2, max_conf = 0.85f),
            ConfirmedItem(class_id = "pasta_box", display_name = "Pasta Box", count = 1, max_conf = 0.92f),
            ConfirmedItem(class_id = "eggs", display_name = "Eggs", count = 12, max_conf = 0.90f)
        )

        viewModel.setInitialInventory(rawItems)
        val items = viewModel.inventory.value

        assertEquals(2, items.size)
        val pasta = items.first { it.class_id == "pasta_noodles" }
        assertEquals(3, pasta.count)
        assertEquals(0.92f, pasta.max_conf, 0.001f)

        val eggs = items.first { it.class_id == "eggs" }
        assertEquals(12, eggs.count)
    }

    @Test
    fun testInventoryViewModelQuantitySteppersAndPruneAtZero() {
        val viewModel = InventoryViewModel()
        viewModel.setInitialInventory(
            listOf(
                ConfirmedItem(class_id = "milk_dairy", display_name = "Milk & Dairy", count = 2, max_conf = 0.8f)
            )
        )

        viewModel.incrementCount("milk_dairy")
        assertEquals(3, viewModel.inventory.value.first().count)

        viewModel.decrementCount("milk_dairy")
        assertEquals(2, viewModel.inventory.value.first().count)

        viewModel.decrementCount("milk_dairy")
        assertEquals(1, viewModel.inventory.value.first().count)

        viewModel.decrementCount("milk_dairy")
        assertTrue(viewModel.inventory.value.isEmpty())
    }

    @Test
    fun testInventoryViewModelRemoveItem() {
        val viewModel = InventoryViewModel()
        viewModel.setInitialInventory(
            listOf(
                ConfirmedItem(class_id = "canned_tomato", display_name = "Canned Tomatoes", count = 4),
                ConfirmedItem(class_id = "cereal_grains", display_name = "Cereal & Grains", count = 1)
            )
        )

        viewModel.removeItem("canned_tomato")
        assertEquals(1, viewModel.inventory.value.size)
        assertEquals("cereal_grains", viewModel.inventory.value.first().class_id)
    }

    @Test
    fun testInventoryViewModelManualAddWithDeduplication() {
        val viewModel = InventoryViewModel()
        viewModel.setInitialInventory(
            listOf(
                ConfirmedItem(class_id = "fresh_produce", display_name = "Fresh Produce", count = 2)
            )
        )

        viewModel.addManualItem(UnifiedItemClass.FRESH_PRODUCE, count = 3)
        assertEquals(1, viewModel.inventory.value.size)
        assertEquals(5, viewModel.inventory.value.first().count)

        viewModel.addManualItem(UnifiedItemClass.COFFEE_CONDIMENT, count = 1)
        assertEquals(2, viewModel.inventory.value.size)
        val coffee = viewModel.inventory.value.first { it.class_id == "coffee_condiment" }
        assertEquals(1, coffee.count)
        assertEquals(1.0f, coffee.max_conf, 0.001f)
    }

    @Test
    fun testInventoryViewModelManualAddAllEightClasses() {
        val viewModel = InventoryViewModel()
        for (entry in UnifiedItemClass.entries) {
            viewModel.addManualItem(entry, 1)
        }
        assertEquals(8, viewModel.inventory.value.size)
    }

    @Test
    fun testInventoryViewModelToItemRequests() {
        val viewModel = InventoryViewModel()
        viewModel.setInitialInventory(
            listOf(
                ConfirmedItem(class_id = "pasta_noodles", display_name = "Pasta & Noodles", count = 2),
                ConfirmedItem(class_id = "eggs", display_name = "Eggs", count = 6)
            )
        )

        val requests = viewModel.toItemRequests()
        val recipeRequests = viewModel.toRecipeInventoryRequests()
        assertEquals(2, requests.size)
        assertEquals(requests, recipeRequests)

        val pastaReq = requests.first { it.class_id == "pasta_noodles" }
        assertEquals("pantry", pastaReq.storage_zone)
        assertEquals(2, pastaReq.count)

        val eggsReq = requests.first { it.class_id == "eggs" }
        assertEquals("refrigerator", eggsReq.storage_zone)
        assertEquals(6, eggsReq.count)
    }

    @Test
    fun testNonExistentItemOperationsDoNotMutatePasta() {
        val viewModel = InventoryViewModel()
        viewModel.setInitialInventory(
            listOf(
                ConfirmedItem(class_id = "pasta_noodles", display_name = "Pasta & Noodles", count = 3),
                ConfirmedItem(class_id = "eggs", display_name = "Eggs", count = 6)
            )
        )

        viewModel.decrementCount("unknown_ghost_item_999")
        val pastaAfterDec = viewModel.inventory.value.first { it.class_id == "pasta_noodles" }
        assertEquals(3, pastaAfterDec.count)

        viewModel.removeItem("unknown_ghost_item_999")
        val pastaExists = viewModel.inventory.value.any { it.class_id == "pasta_noodles" }
        assertTrue(pastaExists)

        viewModel.incrementCount("unknown_ghost_item_999")
        val pastaAfterInc = viewModel.inventory.value.first { it.class_id == "pasta_noodles" }
        assertEquals(3, pastaAfterInc.count)

        val requests = viewModel.toRecipeInventoryRequests()
        assertEquals(2, requests.size)
        val pastaReq = requests.first { it.class_id == "pasta_noodles" }
        assertEquals("pantry", pastaReq.storage_zone)
        assertEquals(3, pastaReq.count)
        val eggsReq = requests.first { it.class_id == "eggs" }
        assertEquals("refrigerator", eggsReq.storage_zone)
        assertEquals(6, eggsReq.count)
    }

    @Test
    fun testInventoryViewModelDialogAndClear() {
        val viewModel = InventoryViewModel()
        assertFalse(viewModel.showAddDialog.value)

        viewModel.openAddDialog()
        assertTrue(viewModel.showAddDialog.value)

        viewModel.closeAddDialog()
        assertFalse(viewModel.showAddDialog.value)

        viewModel.addManualItem(UnifiedItemClass.MILK_DAIRY, 2)
        assertEquals(1, viewModel.inventory.value.size)

        viewModel.clearInventory()
        assertTrue(viewModel.inventory.value.isEmpty())
    }

    @Test
    fun testRecipeViewModelInitialAndIdleState() {
        val viewModel = RecipeViewModel()
        assertEquals(RecipeUiState.Idle, viewModel.uiState.value)
    }

    @Test
    fun testRecipeViewModelSuccessfulGeneration() = runTest {
        val fakeResponse = RecipeGenerateResponse(
            source = "gemini-1.5-flash",
            cached = false,
            latency_ms = 412.5f,
            inventory_used = listOf("pasta_noodles", "canned_tomato"),
            recipes = listOf(
                RecipeCard(
                    id = "recipe_1",
                    title = "Rustic Tomato Pasta",
                    prep_time_mins = 20,
                    difficulty = "Easy",
                    storage_zones_used = listOf("Pantry"),
                    matched_ingredients = listOf("Pasta & Noodles", "Canned Tomatoes"),
                    additional_staples = listOf("Salt", "Olive Oil", "Water"),
                    steps = listOf("Boil salted water", "Cook pasta 8 mins", "Simmer tomato sauce", "Toss together"),
                    zero_waste_tip = "Use reserved pasta water to enrich sauce emulsion."
                ),
                RecipeCard(
                    id = "recipe_2",
                    title = "Quick Tuna Arrabbiata",
                    prep_time_mins = 25,
                    difficulty = "Medium",
                    storage_zones_used = listOf("Pantry"),
                    matched_ingredients = listOf("Pasta & Noodles", "Canned Tomatoes"),
                    additional_staples = listOf("Chili flakes", "Garlic"),
                    steps = listOf("Saute garlic", "Add tomatoes", "Combine pasta"),
                    zero_waste_tip = "Store leftover tomato paste in freezer."
                ),
                RecipeCard(
                    id = "recipe_3",
                    title = "One-Pot Pantry Bake",
                    prep_time_mins = 30,
                    difficulty = "Easy",
                    storage_zones_used = listOf("Pantry"),
                    matched_ingredients = listOf("Pasta & Noodles"),
                    additional_staples = listOf("Cheese", "Black pepper"),
                    steps = listOf("Mix in baking dish", "Bake 20 mins at 180C"),
                    zero_waste_tip = "Use stale bread crusts for crispy topping."
                )
            )
        )

        val fakeApi = object : InferenceApiService {
            override suspend fun getHealth(): HealthResponse = HealthResponse()
            override suspend fun generateRecipes(request: RecipeGenerateRequest): RecipeGenerateResponse = fakeResponse
        }

        val viewModel = RecipeViewModel(apiService = fakeApi)
        val items = listOf(
            InventoryItemRequest(class_id = "pasta_noodles", display_name = "Pasta & Noodles", count = 2)
        )

        viewModel.generateRecipes(items)
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertTrue(state is RecipeUiState.Success)
        val success = state as RecipeUiState.Success
        assertEquals("gemini-1.5-flash", success.response.source)
        assertEquals(3, success.response.recipes.size)
        assertEquals("Rustic Tomato Pasta", success.response.recipes[0].title)
        assertEquals(4, success.response.recipes[0].steps.size)
    }

    @Test
    fun testRecipeViewModelInteractiveStepToggle() = runTest {
        val fakeResponse = RecipeGenerateResponse(
            source = "offline_heuristic",
            cached = true,
            latency_ms = 12.0f,
            recipes = listOf(
                RecipeCard(
                    id = "card_a",
                    title = "Card A",
                    steps = listOf("Step 0", "Step 1", "Step 2")
                ),
                RecipeCard(
                    id = "card_b",
                    title = "Card B",
                    steps = listOf("Step 0", "Step 1")
                )
            )
        )

        val fakeApi = object : InferenceApiService {
            override suspend fun getHealth(): HealthResponse = HealthResponse()
            override suspend fun generateRecipes(request: RecipeGenerateRequest): RecipeGenerateResponse = fakeResponse
        }

        val viewModel = RecipeViewModel(apiService = fakeApi)
        viewModel.generateRecipes(emptyList())
        advanceUntilIdle()

        assertFalse(viewModel.isStepChecked("card_a", 0))
        assertFalse(viewModel.isStepChecked("card_a", 1))

        viewModel.toggleStep("card_a", 0)
        assertTrue(viewModel.isStepChecked("card_a", 0))
        assertFalse(viewModel.isStepChecked("card_a", 1))

        viewModel.toggleStep("card_a", 1)
        assertTrue(viewModel.isStepChecked("card_a", 0))
        assertTrue(viewModel.isStepChecked("card_a", 1))

        assertFalse(viewModel.isStepChecked("card_b", 0))

        viewModel.toggleStep("card_a", 0)
        assertFalse(viewModel.isStepChecked("card_a", 0))
        assertTrue(viewModel.isStepChecked("card_a", 1))
    }

    @Test
    fun testRecipeViewModelErrorAndRetry() = runTest {
        var callCount = 0
        val fakeResponse = RecipeGenerateResponse(
            source = "retry_success",
            recipes = listOf(RecipeCard(id = "r1", title = "Recipe 1"))
        )

        val failingApi = object : InferenceApiService {
            override suspend fun getHealth(): HealthResponse = HealthResponse()
            override suspend fun generateRecipes(request: RecipeGenerateRequest): RecipeGenerateResponse {
                callCount++
                if (callCount == 1) {
                    throw IOException("Network connection timed out")
                }
                return fakeResponse
            }
        }

        val viewModel = RecipeViewModel(apiService = failingApi)
        val items = listOf(
            InventoryItemRequest(class_id = "eggs", display_name = "Eggs", count = 4)
        )

        viewModel.generateRecipes(items)
        advanceUntilIdle()

        val state1 = viewModel.uiState.value
        assertTrue(state1 is RecipeUiState.Error)
        val errorState = state1 as RecipeUiState.Error
        assertTrue(errorState.message.contains("timed out"))
        assertTrue(errorState.canRetry)

        viewModel.retry()
        advanceUntilIdle()

        val state2 = viewModel.uiState.value
        assertTrue(state2 is RecipeUiState.Success)
        val successState = state2 as RecipeUiState.Success
        assertEquals("retry_success", successState.response.source)
        assertEquals(2, callCount)
    }

    @Test
    fun testRecipeViewModelDirectInjectionAndReset() {
        val viewModel = RecipeViewModel()
        val customResponse = RecipeGenerateResponse(
            source = "custom_test",
            cached = false,
            latency_ms = 50.0f,
            recipes = listOf(RecipeCard(id = "inj_1", title = "Injected Recipe"))
        )

        viewModel.injectRecipes(customResponse)
        val state = viewModel.uiState.value
        assertTrue(state is RecipeUiState.Success)
        assertEquals("custom_test", (state as RecipeUiState.Success).response.source)

        viewModel.resetState()
        assertEquals(RecipeUiState.Idle, viewModel.uiState.value)
    }

    @Test
    fun testRecipeSerializationAndDeserialization() {
        val originalCard = RecipeCard(
            id = "rec_001",
            title = "Zero-Waste Frittata",
            prep_time_mins = 15,
            difficulty = "Easy",
            storage_zones_used = listOf("Refrigerator", "Pantry"),
            matched_ingredients = listOf("Eggs", "Milk & Dairy", "Fresh Produce"),
            additional_staples = listOf("Salt", "Black Pepper", "Olive Oil"),
            steps = listOf(
                "Whisk eggs with a splash of milk",
                "Chop leftover produce into bite-sized pieces",
                "Cook in skillet until set"
            ),
            zero_waste_tip = "Include vegetable stems and herb stems finely chopped."
        )

        val originalResponse = RecipeGenerateResponse(
            source = "gemini-1.5-flash",
            cached = true,
            latency_ms = 350.5f,
            inventory_used = listOf("eggs", "milk_dairy", "fresh_produce"),
            recipes = listOf(originalCard)
        )

        val jsonStr = NetworkClientProvider.json.encodeToString(originalResponse)
        assertTrue(jsonStr.contains("Zero-Waste Frittata"))
        assertTrue(jsonStr.contains("gemini-1.5-flash"))

        val decodedResponse = NetworkClientProvider.json.decodeFromString<RecipeGenerateResponse>(jsonStr)
        assertEquals(originalResponse.source, decodedResponse.source)
        assertEquals(originalResponse.cached, decodedResponse.cached)
        assertEquals(originalResponse.latency_ms, decodedResponse.latency_ms, 0.001f)
        assertEquals(1, decodedResponse.recipes.size)

        val decodedCard = decodedResponse.recipes[0]
        assertEquals(originalCard.id, decodedCard.id)
        assertEquals(originalCard.title, decodedCard.title)
        assertEquals(originalCard.prep_time_mins, decodedCard.prep_time_mins)
        assertEquals(originalCard.difficulty, decodedCard.difficulty)
        assertEquals(originalCard.storage_zones_used, decodedCard.storage_zones_used)
        assertEquals(originalCard.matched_ingredients, decodedCard.matched_ingredients)
        assertEquals(originalCard.additional_staples, decodedCard.additional_staples)
        assertEquals(originalCard.steps, decodedCard.steps)
        assertEquals(originalCard.zero_waste_tip, decodedCard.zero_waste_tip)
    }

    @Test
    fun testRecipeGenerateRequestSerialization() {
        val request = RecipeGenerateRequest(
            confirmed_inventory = listOf(
                InventoryItemRequest(
                    class_id = "pasta_noodles",
                    display_name = "Pasta & Noodles",
                    count = 2,
                    storage_zone = "pantry"
                )
            ),
            timeout_sec = 3.5f,
            api_key = "test_key"
        )

        val jsonStr = NetworkClientProvider.json.encodeToString(request)
        assertTrue(jsonStr.contains("pasta_noodles"))
        assertTrue(jsonStr.contains("3.5"))

        val decoded = NetworkClientProvider.json.decodeFromString<RecipeGenerateRequest>(jsonStr)
        assertEquals(1, decoded.confirmed_inventory.size)
        assertEquals("pasta_noodles", decoded.confirmed_inventory[0].class_id)
        assertEquals(3.5f, decoded.timeout_sec, 0.001f)
        assertEquals("test_key", decoded.api_key)
    }
}
