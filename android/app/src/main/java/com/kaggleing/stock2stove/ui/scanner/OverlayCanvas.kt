package com.kaggleing.stock2stove.ui.scanner

import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.unit.dp
import com.kaggleing.stock2stove.data.model.DetectionItem

object OverlayMath {
    val CONFIRMED_COLOR = Color(0xFF28C850)
    val TRACKING_COLOR = Color(0xFFFFB300)

    fun isTemporallyConfirmed(temporallyConfirmed: Boolean, hitCount: Int = 0, thresholdHits: Int = 3): Boolean {
        return temporallyConfirmed || (hitCount >= thresholdHits)
    }

    fun getConfirmationColor(temporallyConfirmed: Boolean, hitCount: Int = 0): Color {
        return if (isTemporallyConfirmed(temporallyConfirmed, hitCount)) CONFIRMED_COLOR else TRACKING_COLOR
    }

    fun mapNormalizedBbox(bboxNorm: List<Float>, canvasWidth: Float, canvasHeight: Float): Rect {
        if (bboxNorm.size < 4) {
            return Rect(0f, 0f, 0f, 0f)
        }
        val x1 = (bboxNorm[0] * canvasWidth).coerceIn(0f, canvasWidth)
        val y1 = (bboxNorm[1] * canvasHeight).coerceIn(0f, canvasHeight)
        val x2 = (bboxNorm[2] * canvasWidth).coerceIn(0f, canvasWidth)
        val y2 = (bboxNorm[3] * canvasHeight).coerceIn(0f, canvasHeight)
        val left = minOf(x1, x2)
        val top = minOf(y1, y2)
        val right = maxOf(x1, x2)
        val bottom = maxOf(y1, y2)
        return Rect(left, top, right, bottom)
    }

    fun mapNormalizedPolygon(polygonNorm: List<List<Float>>, canvasWidth: Float, canvasHeight: Float): List<Offset> {
        val mapped = mutableListOf<Offset>()
        for (point in polygonNorm) {
            if (point.size >= 2) {
                val x = (point[0] * canvasWidth).coerceIn(0f, canvasWidth)
                val y = (point[1] * canvasHeight).coerceIn(0f, canvasHeight)
                mapped.add(Offset(x, y))
            }
        }
        return mapped
    }

    fun formatLabelText(displayName: String, confidence: Float): String {
        val pct = (confidence * 100).toInt().coerceIn(0, 100)
        return "$displayName $pct%"
    }
}

class TemporalConfirmationTracker(
    private val thresholdHits: Int = 3,
    private val windowSize: Int = 5
) {
    private val classHitHistory = mutableMapOf<String, MutableList<Boolean>>()

    fun recordFrame(detections: List<DetectionItem>) {
        val detectedClasses = detections.map { it.class_id }.toSet()
        val allClasses = (classHitHistory.keys + detectedClasses).toSet()
        for (cls in allClasses) {
            val list = classHitHistory.getOrPut(cls) { mutableListOf() }
            list.add(cls in detectedClasses)
            if (list.size > windowSize) {
                list.removeAt(0)
            }
        }
    }

    fun getHitCount(classId: String): Int {
        return classHitHistory[classId]?.count { it } ?: 0
    }

    fun isConfirmed(classId: String): Boolean {
        return getHitCount(classId) >= thresholdHits
    }

    fun calculateConfirmedTally(): Map<String, Int> {
        val tally = mutableMapOf<String, Int>()
        for ((cls, history) in classHitHistory) {
            val hits = history.count { it }
            if (hits >= thresholdHits) {
                tally[cls] = hits
            }
        }
        return tally
    }

    fun reset() {
        classHitHistory.clear()
    }
}

@Composable
fun OverlayCanvas(
    detections: List<DetectionItem>,
    modifier: Modifier = Modifier,
    confirmedColor: Color = OverlayMath.CONFIRMED_COLOR,
    trackingColor: Color = OverlayMath.TRACKING_COLOR
) {
    Canvas(modifier = modifier) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        val strokeWidth = 3.dp.toPx()
        val textPaint = Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 12.dp.toPx()
            typeface = Typeface.DEFAULT_BOLD
            isAntiAlias = true
        }

        for (detection in detections) {
            val color = OverlayMath.getConfirmationColor(detection.temporally_confirmed)
            val labelText = OverlayMath.formatLabelText(detection.display_name, detection.confidence)

            if (detection.polygon_norm != null && detection.polygon_norm.size == 4) {
                val pts = OverlayMath.mapNormalizedPolygon(detection.polygon_norm, canvasWidth, canvasHeight)
                if (pts.size == 4) {
                    val path = Path().apply {
                        moveTo(pts[0].x, pts[0].y)
                        lineTo(pts[1].x, pts[1].y)
                        lineTo(pts[2].x, pts[2].y)
                        lineTo(pts[3].x, pts[3].y)
                        close()
                    }
                    drawPath(path = path, color = color, style = Stroke(width = strokeWidth))
                    drawPath(path = path, color = color.copy(alpha = 0.15f), style = Fill)

                    val minX = pts.minOf { it.x }
                    val minY = pts.minOf { it.y }
                    val textBounds = android.graphics.Rect()
                    textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)
                    val pillWidth = textBounds.width().toFloat() + 24f
                    val pillHeight = textBounds.height().toFloat() + 16f
                    val pillLeft = minX.coerceIn(0f, (canvasWidth - pillWidth).coerceAtLeast(0f))
                    val pillTop = (minY - pillHeight - 6f).coerceIn(0f, (canvasHeight - pillHeight).coerceAtLeast(0f))

                    drawRoundRect(
                        color = color,
                        topLeft = Offset(pillLeft, pillTop),
                        size = Size(pillWidth, pillHeight),
                        cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx())
                    )

                    drawContext.canvas.nativeCanvas.drawText(
                        labelText,
                        pillLeft + 12f,
                        pillTop + pillHeight - 6f,
                        textPaint
                    )
                }
            } else if (detection.bbox_norm.size >= 4) {
                val rect = OverlayMath.mapNormalizedBbox(detection.bbox_norm, canvasWidth, canvasHeight)
                if (rect.width > 0f && rect.height > 0f) {
                    drawRect(
                        color = color,
                        topLeft = Offset(rect.left, rect.top),
                        size = Size(rect.width, rect.height),
                        style = Stroke(width = strokeWidth)
                    )
                    drawRect(
                        color = color.copy(alpha = 0.15f),
                        topLeft = Offset(rect.left, rect.top),
                        size = Size(rect.width, rect.height),
                        style = Fill
                    )

                    val textBounds = android.graphics.Rect()
                    textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)
                    val pillWidth = textBounds.width().toFloat() + 24f
                    val pillHeight = textBounds.height().toFloat() + 16f
                    val pillLeft = rect.left.coerceIn(0f, (canvasWidth - pillWidth).coerceAtLeast(0f))
                    val pillTop = (rect.top - pillHeight - 6f).coerceIn(0f, (canvasHeight - pillHeight).coerceAtLeast(0f))

                    drawRoundRect(
                        color = color,
                        topLeft = Offset(pillLeft, pillTop),
                        size = Size(pillWidth, pillHeight),
                        cornerRadius = CornerRadius(6.dp.toPx(), 6.dp.toPx())
                    )

                    drawContext.canvas.nativeCanvas.drawText(
                        labelText,
                        pillLeft + 12f,
                        pillTop + pillHeight - 6f,
                        textPaint
                    )
                }
            }
        }
    }
}
