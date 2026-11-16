package com.kaggleing.stock2stove.ui.home

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.kaggleing.stock2stove.data.model.ConnectionStatus
import com.kaggleing.stock2stove.data.model.ScanMode

@Composable
fun HomeScreen(
    viewModel: HomeViewModel = viewModel(),
    onStartScan: (String, ScanMode) -> Unit = { _, _ -> },
    onNavigateToDiagnostics: () -> Unit = {},
    onNavigateToRecipes: () -> Unit = {}
) {
    val state by viewModel.uiState.collectAsState()
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Column {
            Text(
                text = "Stock-to-Stove",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Phase 3 CV Inference & Connection Hub",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "Connection Presets",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    FilterChip(
                        selected = state.selectedPresetId == "usb_adb",
                        onClick = { viewModel.selectPreset("usb_adb") },
                        label = { Text("USB adb") }
                    )
                    FilterChip(
                        selected = state.selectedPresetId == "emulator",
                        onClick = { viewModel.selectPreset("emulator") },
                        label = { Text("Emulator") }
                    )
                    FilterChip(
                        selected = state.selectedPresetId == "custom_lan",
                        onClick = { viewModel.selectPreset("custom_lan") },
                        label = { Text("Custom LAN") }
                    )
                }

                if (state.selectedPresetId == "custom_lan") {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        OutlinedTextField(
                            value = state.customHost,
                            onValueChange = { viewModel.setCustomHost(it) },
                            label = { Text("Host IP") },
                            modifier = Modifier.weight(2f),
                            singleLine = true
                        )
                        OutlinedTextField(
                            value = state.customPort.toString(),
                            onValueChange = {
                                val p = it.toIntOrNull() ?: 8000
                                viewModel.setCustomPort(p)
                            },
                            label = { Text("Port") },
                            modifier = Modifier.weight(1f),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            singleLine = true
                        )
                    }
                }

                Text(
                    text = "Active Base URL: ${state.activeBaseUrl}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            ),
            shape = RoundedCornerShape(12.dp),
            border = CardDefaults.outlinedCardBorder()
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Server Health",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )

                    val statusColor = when (state.connectionStatus) {
                        ConnectionStatus.CONNECTED -> Color(0xFF2E7D32)
                        ConnectionStatus.CONNECTING -> Color(0xFFF57F17)
                        ConnectionStatus.ERROR -> Color(0xFFC62828)
                        ConnectionStatus.DISCONNECTED -> Color(0xFF757575)
                    }

                    val statusLabel = when (state.connectionStatus) {
                        ConnectionStatus.CONNECTED -> "Connected"
                        ConnectionStatus.CONNECTING -> "Connecting..."
                        ConnectionStatus.ERROR -> "Error"
                        ConnectionStatus.DISCONNECTED -> "Offline"
                    }

                    Box(
                        modifier = Modifier
                            .background(statusColor.copy(alpha = 0.15f), RoundedCornerShape(8.dp))
                            .border(1.dp, statusColor, RoundedCornerShape(8.dp))
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = statusLabel,
                            color = statusColor,
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = { viewModel.checkHealth() },
                        enabled = !state.isCheckingHealth
                    ) {
                        if (state.isCheckingHealth) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                strokeWidth = 2.dp,
                                color = MaterialTheme.colorScheme.onPrimary
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Checking...")
                        } else {
                            Text("Ping Server")
                        }
                    }

                    if (state.pingLatencyMs != null) {
                        Text(
                            text = "${state.pingLatencyMs} ms",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                if (state.errorMessage != null) {
                    Text(
                        text = state.errorMessage ?: "",
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall
                    )
                }

                if (state.healthResponse != null) {
                    val resp = state.healthResponse!!
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(
                                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                RoundedCornerShape(8.dp)
                            )
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            text = "Service: ${resp.service} (${resp.status})",
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = "Device: ${resp.hardware.device.ifEmpty { resp.hardware.processor }}",
                            style = MaterialTheme.typography.bodySmall
                        )
                        val weights = resp.model_status.active_weights.ifEmpty {
                            if (resp.model_status.yolov8n_loaded) "yolov8n.pt" else "hybrid"
                        }
                        Text(
                            text = "Active Weights: $weights",
                            style = MaterialTheme.typography.bodySmall
                        )
                        Text(
                            text = "Classes: ${resp.supported_classes.size} supported categories",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "Scan Context Mode",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    FilterChip(
                        selected = state.scanMode == ScanMode.PANTRY_SHELF,
                        onClick = { viewModel.setScanMode(ScanMode.PANTRY_SHELF) },
                        label = { Text("Pantry Shelf") }
                    )
                    FilterChip(
                        selected = state.scanMode == ScanMode.REFRIGERATOR,
                        onClick = { viewModel.setScanMode(ScanMode.REFRIGERATOR) },
                        label = { Text("Refrigerator") }
                    )
                }
            }
        }

        Button(
            onClick = { onStartScan(state.activeBaseUrl, state.scanMode) },
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary
            )
        ) {
            Text(
                text = "Enter Live Scanner",
                style = MaterialTheme.typography.titleMedium
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            androidx.compose.material3.OutlinedButton(
                onClick = onNavigateToRecipes,
                modifier = Modifier
                    .weight(1f)
                    .height(44.dp)
            ) {
                Text("Recipes")
            }
            androidx.compose.material3.OutlinedButton(
                onClick = onNavigateToDiagnostics,
                modifier = Modifier
                    .weight(1f)
                    .height(44.dp)
            ) {
                Text("Diagnostics")
            }
        }
    }
}
