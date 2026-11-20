package com.kaggleing.stock2stove.ui.scanner

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.kaggleing.stock2stove.data.model.DetectorDefenseMode
import com.kaggleing.stock2stove.data.model.DiagnosticsData
import java.util.Locale

@Composable
fun DefenseHud(
    selectedMode: DetectorDefenseMode,
    onModeSelected: (DetectorDefenseMode) -> Unit,
    claheEnabled: Boolean,
    onClaheToggled: (Boolean) -> Unit,
    confThreshold: Float,
    onConfThresholdChanged: (Float) -> Unit,
    fps: Float,
    inferenceMs: Float,
    diagnostics: DiagnosticsData,
    modifier: Modifier = Modifier
) {
    val modesScrollState = rememberScrollState()
    val telemetryScrollState = rememberScrollState()

    val effectiveEdgeDensity = if (diagnostics.edge_density_pct > 0f) {
        diagnostics.edge_density_pct
    } else {
        diagnostics.edge_density * 100f
    }

    val effectiveKeypoints = if (diagnostics.keypoints_detected > 0) {
        diagnostics.keypoints_detected
    } else {
        diagnostics.orb_keypoints
    }

    val effectiveInliers = if (diagnostics.inliers > 0) {
        diagnostics.inliers
    } else {
        diagnostics.ransac_inliers
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(Color(0xE8131A17), RoundedCornerShape(16.dp))
            .border(1.dp, Color(0x28FFFFFF), RoundedCornerShape(16.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(modesScrollState),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            for (mode in DetectorDefenseMode.entries) {
                val isSelected = mode == selectedMode
                FilterChip(
                    selected = isSelected,
                    onClick = { onModeSelected(mode) },
                    label = {
                        Text(
                            text = mode.displayLabel,
                            fontSize = 12.sp,
                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                        )
                    },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = Color(0xFF228B5B),
                        selectedLabelColor = Color.White,
                        containerColor = Color(0xFF232E29),
                        labelColor = Color(0xFFD1D5DB)
                    )
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text(
                    text = "CLAHE",
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium
                )
                Switch(
                    checked = claheEnabled,
                    onCheckedChange = onClaheToggled,
                    modifier = Modifier.size(width = 44.dp, height = 24.dp),
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.White,
                        checkedTrackColor = Color(0xFF228B5B),
                        uncheckedThumbColor = Color.LightGray,
                        uncheckedTrackColor = Color(0xFF374151)
                    )
                )
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier
                    .weight(1f, fill = false)
                    .padding(start = 12.dp)
            ) {
                Text(
                    text = "Conf: ${(confThreshold * 100).toInt()}%",
                    color = Color.White,
                    fontSize = 12.sp
                )
                Slider(
                    value = confThreshold,
                    onValueChange = onConfThresholdChanged,
                    valueRange = 0.1f..0.9f,
                    steps = 7,
                    modifier = Modifier.fillMaxWidth(),
                    colors = SliderDefaults.colors(
                        thumbColor = Color(0xFF34D399),
                        activeTrackColor = Color(0xFF228B5B),
                        inactiveTrackColor = Color(0xFF374151)
                    )
                )
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(telemetryScrollState),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            TelemetryBadge(
                label = "FPS",
                value = String.format(Locale.US, "%.1f", fps),
                color = if (fps >= 5f) Color(0xFF34D399) else Color(0xFFFBBF24)
            )
            TelemetryBadge(
                label = "Latency",
                value = String.format(Locale.US, "%.1f ms", inferenceMs),
                color = if (inferenceMs < 200f) Color(0xFF34D399) else Color(0xFFFBBF24)
            )
            TelemetryBadge(
                label = "Edge",
                value = String.format(Locale.US, "%.1f%%", effectiveEdgeDensity),
                color = Color(0xFFD1D5DB)
            )
            TelemetryBadge(
                label = "Keypoints",
                value = effectiveKeypoints.toString(),
                color = Color(0xFFD1D5DB)
            )
            TelemetryBadge(
                label = "Inliers",
                value = effectiveInliers.toString(),
                color = if (effectiveInliers >= 10) Color(0xFF34D399) else Color(0xFFFCA5A5)
            )
        }
    }
}

@Composable
fun TelemetryBadge(
    label: String,
    value: String,
    color: Color = Color.White
) {
    Row(
        modifier = Modifier
            .background(Color(0xFF1F2924), RoundedCornerShape(8.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "$label:",
            fontSize = 11.sp,
            color = Color(0xFF9CA3AF),
            fontWeight = FontWeight.Medium
        )
        Text(
            text = value,
            fontSize = 11.sp,
            color = color,
            fontWeight = FontWeight.Bold
        )
    }
}
