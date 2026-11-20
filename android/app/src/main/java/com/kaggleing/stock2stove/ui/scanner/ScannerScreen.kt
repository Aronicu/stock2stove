package com.kaggleing.stock2stove.ui.scanner

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Kitchen
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.kaggleing.stock2stove.data.model.ConfirmedItem
import com.kaggleing.stock2stove.data.model.ScanMode
import com.kaggleing.stock2stove.data.model.UnifiedItemClass

@Composable
fun ScannerScreen(
    serverWsUrl: String = "ws://127.0.0.1:8000/ws/scan",
    scanMode: ScanMode = ScanMode.PANTRY_SHELF,
    viewModel: ScannerViewModel = viewModel(),
    onReviewInventory: (List<ConfirmedItem>) -> Unit = {},
    onNavigateBack: () -> Unit = {}
) {
    val uiState by viewModel.uiState.collectAsState()
    val chipsScrollState = rememberScrollState()
    val scannedListScrollState = rememberScrollState()

    LaunchedEffect(serverWsUrl, scanMode) {
        viewModel.initScan(serverWsUrl, scanMode)
    }

    Box(modifier = Modifier.fillMaxSize()) {
        CameraPreview(
            modifier = Modifier.fillMaxSize(),
            targetFps = 8,
            onFrameCaptured = { bytes, b64 ->
                viewModel.onFrameCaptured(bytes, b64)
            }
        )

        OverlayCanvas(
            detections = uiState.detections,
            modifier = Modifier.fillMaxSize()
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    IconButton(
                        onClick = onNavigateBack,
                        modifier = Modifier
                            .background(Color(0xAA000000), CircleShape)
                            .size(40.dp)
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }

                    Row(
                        modifier = Modifier
                            .background(Color(0xAA000000), RoundedCornerShape(16.dp))
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(
                                    if (uiState.isConnected) Color(0xFF28C850) else Color(0xFFFF5252),
                                    CircleShape
                                )
                        )
                        Text(
                            text = "${scanMode.displayName} (${if (uiState.isConnected) "Live" else "Offline"})",
                            color = Color.White,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }

                    ProgressRing(
                        progressSec = uiState.progressSec,
                        totalSec = uiState.totalSec,
                        ringSize = 56.dp,
                        strokeWidth = 4.dp
                    )
                }

                DefenseHud(
                    selectedMode = uiState.selectedMode,
                    onModeSelected = { viewModel.setDetectorMode(it) },
                    claheEnabled = uiState.claheEnabled,
                    onClaheToggled = { viewModel.setClaheEnabled(it) },
                    confThreshold = uiState.confThreshold,
                    onConfThresholdChanged = { viewModel.setConfThreshold(it) },
                    fps = uiState.fps,
                    inferenceMs = uiState.inferenceMs,
                    diagnostics = uiState.diagnostics
                )

                if (uiState.errorMessage != null) {
                    Surface(
                        color = Color(0xDDCC2222),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = uiState.errorMessage ?: "",
                            color = Color.White,
                            fontSize = 12.sp,
                            modifier = Modifier.padding(8.dp)
                        )
                    }
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xDD121212), RoundedCornerShape(16.dp))
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Confirmed Inventory (${uiState.confirmedInventory.size})",
                        color = Color.White,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                    if (uiState.isComplete) {
                        Text(
                            text = "Scan Complete",
                            color = Color(0xFF28C850),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                if (uiState.isComplete) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(max = 220.dp)
                            .verticalScroll(scannedListScrollState),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        if (uiState.confirmedInventory.isEmpty()) {
                            Text(
                                text = "No items detected. Tap Reset to scan again.",
                                color = Color.Gray,
                                fontSize = 12.sp,
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                        } else {
                            for (item in uiState.confirmedInventory) {
                                val cropBitmap = remember(item.crop_jpeg_b64) {
                                    CameraFrameProcessor.decodeBase64Bitmap(item.crop_jpeg_b64)
                                }
                                val displayName = item.display_name.ifBlank {
                                    UnifiedItemClass.normalize(item.class_id).displayName
                                }
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .background(Color(0xFF1E1E1E), RoundedCornerShape(10.dp))
                                        .padding(8.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                                ) {
                                    if (cropBitmap != null) {
                                        Image(
                                            bitmap = cropBitmap.asImageBitmap(),
                                            contentDescription = displayName,
                                            contentScale = ContentScale.Crop,
                                            modifier = Modifier
                                                .size(48.dp)
                                                .clip(RoundedCornerShape(8.dp))
                                        )
                                    } else {
                                        Box(
                                            modifier = Modifier
                                                .size(48.dp)
                                                .clip(RoundedCornerShape(8.dp))
                                                .background(Color(0xFF2C2C2C)),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Kitchen,
                                                contentDescription = displayName,
                                                tint = Color.Gray,
                                                modifier = Modifier.size(20.dp)
                                            )
                                        }
                                    }

                                    Column(
                                        modifier = Modifier.weight(1f),
                                        verticalArrangement = Arrangement.spacedBy(2.dp)
                                    ) {
                                        Text(
                                            text = displayName,
                                            color = Color.White,
                                            fontSize = 13.sp,
                                            fontWeight = FontWeight.SemiBold
                                        )
                                        if (item.max_conf > 0.0f) {
                                            Text(
                                                text = "${(item.max_conf * 100).toInt()}% conf",
                                                color = Color(0xFFB9F6CA),
                                                fontSize = 11.sp
                                            )
                                        }
                                    }

                                    Box(
                                        modifier = Modifier
                                            .background(Color(0xFF1B5E20), RoundedCornerShape(6.dp))
                                            .padding(horizontal = 8.dp, vertical = 4.dp)
                                    ) {
                                        Text(
                                            text = "x${item.count}",
                                            color = Color(0xFFB9F6CA),
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }
                        }
                    }
                } else {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(chipsScrollState),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        if (uiState.confirmedInventory.isEmpty()) {
                            Text(
                                text = "Pan slowly across shelves or fridge...",
                                color = Color.Gray,
                                fontSize = 12.sp,
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                        } else {
                            for (item in uiState.confirmedInventory) {
                                val cropBitmap = remember(item.crop_jpeg_b64) {
                                    CameraFrameProcessor.decodeBase64Bitmap(item.crop_jpeg_b64)
                                }
                                FilterChip(
                                    selected = true,
                                    onClick = {},
                                    leadingIcon = if (cropBitmap != null) {
                                        {
                                            Image(
                                                bitmap = cropBitmap.asImageBitmap(),
                                                contentDescription = item.display_name,
                                                contentScale = ContentScale.Crop,
                                                modifier = Modifier
                                                    .size(24.dp)
                                                    .clip(CircleShape)
                                            )
                                        }
                                    } else {
                                        null
                                    },
                                    label = {
                                        Text(
                                            text = "${item.display_name} x${item.count}",
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.SemiBold
                                        )
                                    },
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = Color(0xFF1B5E20),
                                        selectedLabelColor = Color(0xFFB9F6CA)
                                    )
                                )
                            }
                        }
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(
                        onClick = { viewModel.togglePause() },
                        modifier = Modifier
                            .background(Color(0xFF2A2A2A), RoundedCornerShape(8.dp))
                            .size(44.dp)
                    ) {
                        Icon(
                            imageVector = if (uiState.isPaused) Icons.Default.PlayArrow else Icons.Default.Pause,
                            contentDescription = if (uiState.isPaused) "Resume" else "Pause",
                            tint = Color.White
                        )
                    }

                    IconButton(
                        onClick = { viewModel.resetScan() },
                        modifier = Modifier
                            .background(Color(0xFF2A2A2A), RoundedCornerShape(8.dp))
                            .size(44.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Reset",
                            tint = Color.White
                        )
                    }

                    Button(
                        onClick = { onReviewInventory(uiState.confirmedInventory) },
                        modifier = Modifier
                            .weight(1f)
                            .height(44.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (uiState.isComplete || uiState.confirmedInventory.isNotEmpty()) {
                                MaterialTheme.colorScheme.primary
                            } else {
                                Color(0xFF333333)
                            }
                        ),
                        enabled = uiState.isComplete || uiState.confirmedInventory.isNotEmpty()
                    ) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = "Review",
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "Review Inventory",
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}
