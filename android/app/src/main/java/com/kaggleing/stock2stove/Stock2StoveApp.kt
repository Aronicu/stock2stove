package com.kaggleing.stock2stove

import android.net.Uri
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.kaggleing.stock2stove.data.repository.SharedPrefsServerConfigStore
import com.kaggleing.stock2stove.ui.diagnostics.DiagnosticsScreen
import com.kaggleing.stock2stove.ui.home.HomeScreen
import com.kaggleing.stock2stove.ui.home.HomeViewModel
import com.kaggleing.stock2stove.ui.inventory.InventoryBottomSheet
import com.kaggleing.stock2stove.ui.inventory.InventoryViewModel
import com.kaggleing.stock2stove.ui.recipes.RecipeScreen
import com.kaggleing.stock2stove.ui.recipes.RecipeViewModel
import com.kaggleing.stock2stove.ui.scanner.ScannerScreen
import com.kaggleing.stock2stove.ui.scanner.ScannerViewModel

@Composable
fun Stock2StoveApp(
    inventoryViewModel: InventoryViewModel = viewModel(),
    recipeViewModel: RecipeViewModel = viewModel()
) {
    val context = LocalContext.current.applicationContext
    val homeFactory = remember(context) {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                return HomeViewModel(
                    configStore = SharedPrefsServerConfigStore(context)
                ) as T
            }
        }
    }
    val homeViewModel: HomeViewModel = viewModel(factory = homeFactory)
    val navController = rememberNavController()
    var showInventorySheet by remember { mutableStateOf(false) }
    val basketItems by inventoryViewModel.inventory.collectAsState()

    NavHost(
        navController = navController,
        startDestination = "home"
    ) {
        composable("home") {
            HomeScreen(
                viewModel = homeViewModel,
                basketItems = basketItems,
                onStartScan = { baseUrl ->
                    val encodedUrl = Uri.encode(baseUrl)
                    navController.navigate("scanner?serverUrl=$encodedUrl")
                },
                onGenerateRecipesFromBasket = {
                    val requests = inventoryViewModel.toItemRequests()
                    if (requests.isNotEmpty()) {
                        recipeViewModel.generateRecipes(requests)
                        navController.navigate("recipes")
                    }
                },
                onClearBasket = {
                    inventoryViewModel.clearInventory()
                },
                onNavigateToDiagnostics = {
                    navController.navigate("diagnostics")
                },
                onNavigateToRecipes = {
                    navController.navigate("recipes")
                }
            )
        }

        composable(
            route = "scanner?serverUrl={serverUrl}",
            arguments = listOf(
                navArgument("serverUrl") {
                    type = NavType.StringType
                    defaultValue = "http://192.168.1.100:8000"
                }
            )
        ) { backStackEntry ->
            val serverUrlStr = backStackEntry.arguments?.getString("serverUrl") ?: "http://192.168.1.100:8000"
            val decodedUrl = Uri.decode(serverUrlStr)

            val wsUrl = if (decodedUrl.startsWith("http://")) {
                decodedUrl.replace("http://", "ws://") + "/ws/scan"
            } else if (decodedUrl.startsWith("https://")) {
                decodedUrl.replace("https://", "wss://") + "/ws/scan"
            } else {
                "ws://$decodedUrl/ws/scan"
            }

            val scannerViewModel: ScannerViewModel = viewModel()

            ScannerScreen(
                serverWsUrl = wsUrl,
                viewModel = scannerViewModel,
                initialBasket = basketItems,
                onInventoryChanged = { items ->
                    inventoryViewModel.replaceInventory(items)
                },
                onReviewInventory = { confirmedItems ->
                    inventoryViewModel.replaceInventory(confirmedItems)
                    showInventorySheet = true
                },
                onGenerateRecipesNow = { confirmedItems ->
                    inventoryViewModel.replaceInventory(confirmedItems)
                    val requests = inventoryViewModel.toItemRequests()
                    if (requests.isNotEmpty()) {
                        recipeViewModel.generateRecipes(requests)
                        navController.navigate("recipes")
                    }
                },
                onNavigateBack = {
                    navController.popBackStack()
                }
            )

            if (showInventorySheet) {
                InventoryBottomSheet(
                    viewModel = inventoryViewModel,
                    onDismiss = {
                        scannerViewModel.syncConfirmedInventory(inventoryViewModel.inventory.value)
                        showInventorySheet = false
                    },
                    onScanMore = {
                        scannerViewModel.syncConfirmedInventory(inventoryViewModel.inventory.value)
                        showInventorySheet = false
                        scannerViewModel.retryScan()
                    },
                    onClearAll = {
                        inventoryViewModel.clearInventory()
                        scannerViewModel.clearInventory()
                    },
                    onGenerateRecipes = { itemRequests ->
                        scannerViewModel.syncConfirmedInventory(inventoryViewModel.inventory.value)
                        showInventorySheet = false
                        recipeViewModel.generateRecipes(itemRequests)
                        navController.navigate("recipes")
                    }
                )
            }
        }

        composable("recipes") {
            RecipeScreen(
                viewModel = recipeViewModel,
                onScanMore = {
                    navController.popBackStack()
                },
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }

        composable("diagnostics") {
            DiagnosticsScreen(
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }
    }
}
