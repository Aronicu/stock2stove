package com.kaggleing.stock2stove.ui.scanner

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Kitchen
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.kaggleing.stock2stove.data.model.ConfirmedItem
import com.kaggleing.stock2stove.data.model.UnifiedItemClass

@Composable
fun ScannerScreen(
    serverWsUrl: String = "ws://127.0.0.1:8000/ws/scan",
    viewModel: ScannerViewModel = viewModel(),
    initialBasket: List<ConfirmedItem> = emptyList(),
    onInventoryChanged: (List<ConfirmedItem>) -> Unit = {},
    onReviewInventory: (List<ConfirmedItem>) -> Unit = {},
    onGenerateRecipesNow: (List<ConfirmedItem>) -> Unit = {},
    onNavigateBack: () -> Unit = {}
) {
    val uiState by viewModel.uiState.collectAsState()
    val chipsScrollState = rememberScrollState()
    val scannedListScrollState = rememberScrollState()
    var showHudControls by remember { mutableStateOf(false) }
    var showClearConfirmDialog by remember { mutableStateOf(false) }
    var hasSeeded by remember { mutableStateOf(false) }

    LaunchedEffect(serverWsUrl) {
        viewModel.seedInventoryIfEmpty(initialBasket)
        hasSeeded = true
        viewModel.initScan(serverWsUrl)
    }

    LaunchedEffect(hasSeeded, uiState.confirmedInventory) {
        if (hasSeeded) {
            onInventoryChanged(uiState.confirmedInventory)
        }
    }

    if (showClearConfirmDialog) {
        AlertDialog(
            onDismissRequest = { showClearConfirmDialog = false },
            title = { Text("Clear Scanned Items?") },
            text = {
                Text("This will remove all ${uiState.confirmedInventory.size} detected food items.")
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showClearConfirmDialog = false
                        viewModel.clearInventory()
                    }
                ) {
                    Text("Clear All", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirmDialog = false }) {
                    Text("Keep Items")
                }
            }
        )
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
                .statusBarsPadding()
                .navigationBarsPadding()
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
                            .background(Color(0xB8111915), CircleShape)
                            .size(42.dp)
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }

                    Row(
                        modifier = Modifier
                            .background(Color(0xB8111915), RoundedCornerShape(20.dp))
                            .border(1.dp, Color(0x33FFFFFF), RoundedCornerShape(20.dp))
                            .padding(horizontal = 12.dp, vertical = 7.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(
                                    if (uiState.isConnected) Color(0xFF34D399) else Color(0xFFF87171),
                                    CircleShape
                                )
                        )
                        Text(
                            text = "${uiState.selectedMode.displayLabel} · ${if (uiState.isConnected) "Live" else "Offline"}",
                            color = Color.White,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        IconButton(
                            onClick = { showHudControls = !showHudControls },
                            modifier = Modifier
                                .background(
                                    if (showHudControls) Color(0xFF1B5E3F) else Color(0xB8111915),
                                    CircleShape
                                )
                                .size(40.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Tune,
                                contentDescription = "Detector Settings",
                                tint = Color.White,
                                modifier = Modifier.size(20.dp)
                            )
                        }

                        ProgressRing(
                            progressSec = uiState.progressSec,
                            totalSec = uiState.totalSec,
                            ringSize = 54.dp,
                            strokeWidth = 4.dp
                        )
                    }
                }

                AnimatedVisibility(visible = showHudControls) {
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
                }

                if (uiState.errorMessage != null) {
                    Surface(
                        color = Color(0xE0B91C1C),
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = uiState.errorMessage ?: "",
                            color = Color.White,
                            fontSize = 12.sp,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)
                        )
                    }
                }
            }

            if (!uiState.isComplete && uiState.isAnalyzingFrame) {
                Row(
                    modifier = Modifier
                        .align(Alignment.CenterHorizontally)
                        .background(Color(0xC4111915), RoundedCornerShape(20.dp))
                        .padding(horizontal = 14.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(14.dp),
                        strokeWidth = 2.dp,
                        color = Color(0xFF6EE7B7)
                    )
                    Text(
                        text = "Gemini Vision analyzing...",
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xEB131A17), RoundedCornerShape(24.dp))
                    .border(1.dp, Color(0x28FFFFFF), RoundedCornerShape(24.dp))
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (!uiState.isComplete) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "Detected Food (${uiState.confirmedInventory.size})",
                            color = Color.White,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold
                        )
                        if (uiState.confirmedInventory.isNotEmpty()) {
                            TextButton(
                                onClick = { showClearConfirmDialog = true },
                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                            ) {
                                Text(
                                    text = "Clear",
                                    color = Color(0xFFFCA5A5),
                                    fontSize = 12.sp
                                )
                            }
                        }
                    }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .horizontalScroll(chipsScrollState),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        if (uiState.confirmedInventory.isEmpty()) {
                            Text(
                                text = "Pan slowly across your pantry or fridge...",
                                color = Color(0xFF9CA3AF),
                                fontSize = 12.sp,
                                modifier = Modifier.padding(vertical = 4.dp)
                            )
                        } else {
                            for (item in uiState.confirmedInventory) {
                                val label = item.display_name.ifBlank {
                                    UnifiedItemClass.normalize(item.class_id).displayName
                                }
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
                                                contentDescription = label,
                                                contentScale = ContentScale.Crop,
                                                modifier = Modifier
                                                    .size(22.dp)
                                                    .clip(CircleShape)
                                            )
                                        }
                                    } else {
                                        null
                                    },
                                    label = {
                                        Text(
                                            text = "$label x${item.count}",
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.SemiBold
                                        )
                                    },
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = Color(0xFF1B5E3F),
                                        selectedLabelColor = Color(0xFFD7F2E3)
                                    )
                                )
                            }
                        }
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        IconButton(
                            onClick = { viewModel.togglePause() },
                            modifier = Modifier
                                .background(Color(0xFF232E29), RoundedCornerShape(12.dp))
                                .size(46.dp)
                        ) {
                            Icon(
                                imageVector = if (uiState.isPaused) Icons.Default.PlayArrow else Icons.Default.Pause,
                                contentDescription = if (uiState.isPaused) "Resume" else "Pause",
                                tint = Color.White
                            )
                        }

                        IconButton(
                            onClick = { viewModel.retryScan() },
                            modifier = Modifier
                                .background(Color(0xFF232E29), RoundedCornerShape(12.dp))
                                .size(46.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Refresh,
                                contentDescription = "Retry Scan",
                                tint = Color.White
                            )
                        }

                        Button(
                            onClick = { viewModel.triggerComplete() },
                            enabled = uiState.confirmedInventory.isNotEmpty(),
                            modifier = Modifier
                                .weight(1f)
                                .height(46.dp),
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF228B5B),
                                disabledContainerColor = Color(0xFF26302B)
                            )
                        ) {
                            Icon(
                                imageVector = Icons.Default.Check,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = "Done Scanning",
                                fontSize = 13.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                } else {
                    if (uiState.confirmedInventory.isNotEmpty()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Text(
                                    text = "Food Detected (${uiState.confirmedInventory.size})",
                                    color = Color.White,
                                    fontSize = 15.sp,
                                    fontWeight = FontWeight.Bold
                                )
                                Box(
                                    modifier = Modifier
                                        .background(Color(0xFF1B5E3F), RoundedCornerShape(8.dp))
                                        .padding(horizontal = 8.dp, vertical = 2.dp)
                                ) {
                                    Text(
                                        text = "Ready",
                                        color = Color(0xFFA7F3D0),
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }

                            TextButton(
                                onClick = { showClearConfirmDialog = true },
                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                            ) {
                                Text(
                                    text = "Clear",
                                    color = Color(0xFFFCA5A5),
                                    fontSize = 12.sp
                                )
                            }
                        }

                        Text(
                            text = "Do you still need to scan more food?",
                            color = Color(0xFFA7F3D0),
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold
                        )

                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(max = 200.dp)
                                .verticalScroll(scannedListScrollState),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
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
                                        .background(Color(0xFF1D2622), RoundedCornerShape(12.dp))
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
                                                .size(44.dp)
                                                .clip(RoundedCornerShape(8.dp))
                                        )
                                    } else {
                                        Box(
                                            modifier = Modifier
                                                .size(44.dp)
                                                .clip(RoundedCornerShape(8.dp))
                                                .background(Color(0xFF28342E)),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Icon(
                                                imageVector = Icons.Default.Kitchen,
                                                contentDescription = displayName,
                                                tint = Color(0xFF9CA3AF),
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
                                            fontWeight = FontWeight.SemiBold,
                                            maxLines = 1,
                                            overflow = TextOverflow.Ellipsis
                                        )
                                        if (item.max_conf > 0.0f) {
                                            Text(
                                                text = "${(item.max_conf * 100).toInt()}% conf",
                                                color = Color(0xFF6EE7B7),
                                                fontSize = 11.sp
                                            )
                                        }
                                    }

                                    Box(
                                        modifier = Modifier
                                            .background(Color(0xFF1B5E3F), RoundedCornerShape(8.dp))
                                            .padding(horizontal = 8.dp, vertical = 4.dp)
                                    ) {
                                        Text(
                                            text = "x${item.count}",
                                            color = Color(0xFFD7F2E3),
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }

                                    IconButton(
                                        onClick = { viewModel.removeConfirmedItem(item.class_id) },
                                        modifier = Modifier.size(28.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Close,
                                            contentDescription = "Remove",
                                            tint = Color(0xFF9CA3AF),
                                            modifier = Modifier.size(16.dp)
                                        )
                                    }
                                }
                            }
                        }

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            OutlinedButton(
                                onClick = { viewModel.retryScan() },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(48.dp),
                                shape = RoundedCornerShape(12.dp),
                                border = BorderStroke(1.dp, Color(0xFF6EE7B7)),
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = Color(0xFFA7F3D0)
                                )
                            ) {
                                Icon(
                                    imageVector = Icons.Default.AddAPhoto,
                                    contentDescription = null,
                                    modifier = Modifier.size(17.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "Scan More",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }

                            Button(
                                onClick = { onGenerateRecipesNow(uiState.confirmedInventory) },
                                modifier = Modifier
                                    .weight(1.15f)
                                    .height(48.dp),
                                shape = RoundedCornerShape(12.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color(0xFF228B5B),
                                    contentColor = Color.White
                                )
                            ) {
                                Icon(
                                    imageVector = Icons.Default.AutoAwesome,
                                    contentDescription = null,
                                    modifier = Modifier.size(17.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "Generate Recipes",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }

                        TextButton(
                            onClick = { onReviewInventory(uiState.confirmedInventory) },
                            modifier = Modifier.align(Alignment.CenterHorizontally)
                        ) {
                            Text(
                                text = "Edit Quantities or Add Manually",
                                color = Color(0xFFD1D5DB),
                                fontSize = 12.sp
                            )
                        }
                    } else {
                        Text(
                            text = "No Food Detected Yet",
                            color = Color.White,
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Hold steady with clear lighting and try again, or add items manually.",
                            color = Color(0xFF9CA3AF),
                            fontSize = 12.sp
                        )

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            OutlinedButton(
                                onClick = { viewModel.retryScan() },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(46.dp),
                                shape = RoundedCornerShape(12.dp),
                                border = BorderStroke(1.dp, Color(0xFF6EE7B7)),
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = Color(0xFFA7F3D0)
                                )
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Refresh,
                                    contentDescription = null,
                                    modifier = Modifier.size(17.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = "Retry Scan",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }

                            Button(
                                onClick = { onReviewInventory(emptyList()) },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(46.dp),
                                shape = RoundedCornerShape(12.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Color(0xFF228B5B),
                                    contentColor = Color.White
                                )
                            ) {
                                Text(
                                    text = "Add Manually",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}
