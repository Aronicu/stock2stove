package com.kaggleing.stock2stove.ui.scanner

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import java.util.Locale

object ProgressRingMath {
    fun calculateProgressFraction(currentSec: Float, totalSec: Float = 5.0f): Float {
        if (totalSec <= 0f) return 1.0f
        return (currentSec / totalSec).coerceIn(0.0f, 1.0f)
    }

    fun calculateSweepAngle(currentSec: Float, totalSec: Float = 5.0f): Float {
        return calculateProgressFraction(currentSec, totalSec) * 360f
    }

    fun formatTimerText(currentSec: Float, totalSec: Float = 5.0f): String {
        val clampedCurrent = currentSec.coerceIn(0.0f, totalSec)
        return String.format(Locale.US, "%.1fs / %.1fs", clampedCurrent, totalSec)
    }

    fun isCountdownComplete(currentSec: Float, totalSec: Float = 5.0f): Boolean {
        return currentSec >= totalSec
    }
}

@Composable
fun ProgressRing(
    progressSec: Float,
    totalSec: Float = 5.0f,
    modifier: Modifier = Modifier,
    ringSize: Dp = 80.dp,
    strokeWidth: Dp = 6.dp,
    activeColor: Color = Color(0xFF28C850),
    trackColor: Color = Color.White.copy(alpha = 0.25f)
) {
    val sweepAngle = ProgressRingMath.calculateSweepAngle(progressSec, totalSec)
    val timerString = ProgressRingMath.formatTimerText(progressSec, totalSec)

    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier.size(ringSize)
    ) {
        Canvas(modifier = Modifier.size(ringSize)) {
            val strokePx = strokeWidth.toPx()
            drawArc(
                color = trackColor,
                startAngle = 0f,
                sweepAngle = 360f,
                useCenter = false,
                style = Stroke(width = strokePx)
            )
            drawArc(
                color = activeColor,
                startAngle = -90f,
                sweepAngle = sweepAngle,
                useCenter = false,
                style = Stroke(width = strokePx, cap = StrokeCap.Round)
            )
        }

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(4.dp)
        ) {
            Text(
                text = timerString,
                style = MaterialTheme.typography.labelSmall,
                color = Color.White,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
