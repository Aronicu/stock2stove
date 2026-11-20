package com.kaggleing.stock2stove.ui.inventory

import androidx.lifecycle.ViewModel
import com.kaggleing.stock2stove.data.model.ConfirmedItem
import com.kaggleing.stock2stove.data.model.InventoryItemRequest
import com.kaggleing.stock2stove.data.model.UnifiedItemClass
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class InventoryViewModel : ViewModel() {
    private val _inventory = MutableStateFlow<List<ConfirmedItem>>(emptyList())
    val inventory: StateFlow<List<ConfirmedItem>> = _inventory.asStateFlow()

    private val _showAddDialog = MutableStateFlow(false)
    val showAddDialog: StateFlow<Boolean> = _showAddDialog.asStateFlow()

    fun setInitialInventory(items: List<ConfirmedItem>) {
        val merged = mutableMapOf<String, ConfirmedItem>()
        for (item in items) {
            val canonicalId = UnifiedItemClass.normalizeClassId(item.class_id)
            val existing = merged[canonicalId]
            if (existing != null) {
                val combinedCount = existing.count + item.count
                val maxConf = maxOf(existing.max_conf, item.max_conf)
                val bestCrop = if (!item.crop_jpeg_b64.isNullOrBlank() && (existing.crop_jpeg_b64.isNullOrBlank() || item.max_conf >= existing.max_conf)) {
                    item.crop_jpeg_b64
                } else {
                    existing.crop_jpeg_b64
                }
                merged[canonicalId] = existing.copy(
                    count = combinedCount,
                    max_conf = maxConf,
                    crop_jpeg_b64 = bestCrop
                )
            } else {
                val displayName = if (item.display_name.isNotBlank()) {
                    item.display_name
                } else {
                    UnifiedItemClass.normalize(item.class_id).displayName
                }
                merged[canonicalId] = item.copy(class_id = canonicalId, display_name = displayName)
            }
        }
        _inventory.value = merged.values.toList()
    }

    fun replaceInventory(items: List<ConfirmedItem>) {
        val copyList = mutableListOf<ConfirmedItem>()
        for (item in items) {
            copyList.add(item)
        }
        _inventory.value = copyList
    }

    fun incrementCount(classId: String) {
        val targetClass = UnifiedItemClass.fromClassId(classId)
        val current = _inventory.value.toMutableList()
        val index = current.indexOfFirst {
            it.class_id == classId || (targetClass != null && UnifiedItemClass.fromClassId(it.class_id) == targetClass)
        }
        if (index >= 0) {
            val item = current[index]
            current[index] = item.copy(count = item.count + 1)
            _inventory.value = current
        }
    }

    fun decrementCount(classId: String) {
        val targetClass = UnifiedItemClass.fromClassId(classId)
        val current = _inventory.value.toMutableList()
        val index = current.indexOfFirst {
            it.class_id == classId || (targetClass != null && UnifiedItemClass.fromClassId(it.class_id) == targetClass)
        }
        if (index >= 0) {
            val item = current[index]
            if (item.count > 1) {
                current[index] = item.copy(count = item.count - 1)
            } else {
                current.removeAt(index)
            }
            _inventory.value = current
        }
    }

    fun removeItem(classId: String) {
        val targetClass = UnifiedItemClass.fromClassId(classId)
        val current = _inventory.value.toMutableList()
        current.removeAll {
            it.class_id == classId || (targetClass != null && UnifiedItemClass.fromClassId(it.class_id) == targetClass)
        }
        _inventory.value = current
    }

    fun addManualItem(itemClass: UnifiedItemClass, count: Int = 1) {
        if (count <= 0) return
        val current = _inventory.value.toMutableList()
        val index = current.indexOfFirst { it.class_id == itemClass.classId }
        if (index >= 0) {
            val item = current[index]
            current[index] = item.copy(count = item.count + count)
        } else {
            current.add(
                ConfirmedItem(
                    class_id = itemClass.classId,
                    display_name = itemClass.displayName,
                    count = count,
                    max_conf = 1.0f
                )
            )
        }
        _inventory.value = current
    }

    fun addManualItem(classId: String, displayName: String = "", count: Int = 1) {
        if (count <= 0) return
        val unified = UnifiedItemClass.normalize(classId)
        val name = if (displayName.isNotBlank()) displayName else unified.displayName
        val current = _inventory.value.toMutableList()
        val index = current.indexOfFirst { it.class_id == unified.classId }
        if (index >= 0) {
            val item = current[index]
            current[index] = item.copy(count = item.count + count)
        } else {
            current.add(
                ConfirmedItem(
                    class_id = unified.classId,
                    display_name = name,
                    count = count,
                    max_conf = 1.0f
                )
            )
        }
        _inventory.value = current
    }

    fun openAddDialog() {
        _showAddDialog.value = true
    }

    fun closeAddDialog() {
        _showAddDialog.value = false
    }

    fun toItemRequests(): List<InventoryItemRequest> {
        val requests = mutableListOf<InventoryItemRequest>()
        for (item in _inventory.value) {
            val unified = UnifiedItemClass.normalize(item.class_id)
            val zoneKey = unified.defaultStorageZone.zoneKey.lowercase()
            requests.add(
                InventoryItemRequest(
                    class_id = item.class_id,
                    display_name = item.display_name,
                    count = item.count,
                    storage_zone = zoneKey,
                    crop_jpeg_b64 = item.crop_jpeg_b64
                )
            )
        }
        return requests
    }

    fun toRecipeInventoryRequests(): List<InventoryItemRequest> = toItemRequests()

    fun clearInventory() {
        _inventory.value = emptyList()
    }
}
