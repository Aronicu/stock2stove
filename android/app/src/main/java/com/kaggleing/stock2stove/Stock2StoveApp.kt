package com.kaggleing.stock2stove

import android.net.Uri
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.kaggleing.stock2stove.data.model.ScanMode
import com.kaggleing.stock2stove.ui.diagnostics.DiagnosticsScreen
import com.kaggleing.stock2stove.ui.home.HomeScreen
import com.kaggleing.stock2stove.ui.inventory.InventoryBottomSheet
import com.kaggleing.stock2stove.ui.inventory.InventoryViewModel
import com.kaggleing.stock2stove.ui.recipes.RecipeScreen
import com.kaggleing.stock2stove.ui.recipes.RecipeViewModel
import com.kaggleing.stock2stove.ui.scanner.ScannerScreen

@Composable
fun Stock2StoveApp(
    inventoryViewModel: InventoryViewModel = viewModel(),
    recipeViewModel: RecipeViewModel = viewModel()
) {
    val navController = rememberNavController()
    var showInventorySheet by remember { mutableStateOf(false) }

    NavHost(
        navController = navController,
        startDestination = "home"
    ) {
        composable("home") {
            HomeScreen(
                onStartScan = { baseUrl, scanMode ->
                    val encodedUrl = Uri.encode(baseUrl)
                    val modeKey = scanMode.contextKey
                    navController.navigate("scanner?mode=$modeKey&serverUrl=$encodedUrl")
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
            route = "scanner?mode={mode}&serverUrl={serverUrl}",
            arguments = listOf(
                navArgument("mode") {
                    type = NavType.StringType
                    defaultValue = "pantry"
                },
                navArgument("serverUrl") {
                    type = NavType.StringType
                    defaultValue = "http://127.0.0.1:8000"
                }
            )
        ) { backStackEntry ->
            val modeStr = backStackEntry.arguments?.getString("mode") ?: "pantry"
            val serverUrlStr = backStackEntry.arguments?.getString("serverUrl") ?: "http://127.0.0.1:8000"
            val decodedUrl = Uri.decode(serverUrlStr)

            val scanMode = if (modeStr == "fridge") ScanMode.REFRIGERATOR else ScanMode.PANTRY_SHELF
            val wsUrl = if (decodedUrl.startsWith("http://")) {
                decodedUrl.replace("http://", "ws://") + "/ws/scan"
            } else if (decodedUrl.startsWith("https://")) {
                decodedUrl.replace("https://", "wss://") + "/ws/scan"
            } else {
                "ws://$decodedUrl/ws/scan"
            }

            ScannerScreen(
                serverWsUrl = wsUrl,
                scanMode = scanMode,
                onReviewInventory = { confirmedItems ->
                    inventoryViewModel.setInitialInventory(confirmedItems)
                    showInventorySheet = true
                },
                onNavigateBack = {
                    navController.popBackStack()
                }
            )

            if (showInventorySheet) {
                InventoryBottomSheet(
                    viewModel = inventoryViewModel,
                    onDismiss = {
                        showInventorySheet = false
                    },
                    onGenerateRecipes = { itemRequests ->
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
