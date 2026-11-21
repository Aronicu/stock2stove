import { useState } from 'react'
import {
  Activity,
  Cpu,
  Database,
  RefreshCw,
  Server,
  Trash2,
  X,
} from 'lucide-react'
import { getCaloriesPerUnit } from '../constants/catalog.js'

function StatusFlag({ label, ok }) {
  return (
    <div className="flag-row">
      <span>{label}</span>
      <span className={`flag-pill ${ok ? 'flag-ok' : 'flag-off'}`}>
        {ok ? 'Ready' : 'Unavailable'}
      </span>
    </div>
  )
}

export default function DiagnosticsModal({
  isOpen,
  healthState,
  logs,
  onClose,
  onRefreshHealth,
  onClearLogs,
}) {
  const [logFilter, setLogFilter] = useState('ALL')

  if (!isOpen) {
    return null
  }

  const data = healthState.data || {}
  const modelStatus = data.model_status || {}
  const geminiStatus = modelStatus.gemini_status || {}
  const hardware = data.hardware || {}
  const modes = Array.isArray(data.supported_detector_modes) ? data.supported_detector_modes : []
  const classes = Array.isArray(data.supported_classes) ? data.supported_classes : []

  const filteredLogs = logs.filter(function matchLogFilter(entry) {
    if (logFilter === 'ALL') {
      return true
    }
    return String(entry.level).toUpperCase() === logFilter
  })

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="diag-modal-title">
      <div className="modal-card modal-card-lg">
        <div className="modal-header">
          <div className="diag-title-group">
            <Activity size={20} />
            <div>
              <h3 id="diag-modal-title">Server Health &amp; Diagnostics</h3>
              <p className="helper-text">
                {data.service || 'stock2stove-inference-server'} · {healthState.status === 'connected' ? `Online (${healthState.latencyMs} ms)` : 'Checking'}
              </p>
            </div>
          </div>

          <div className="action-row">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={onRefreshHealth}
            >
              <RefreshCw size={15} />
              <span>Refresh</span>
            </button>
            <button
              type="button"
              className="btn-close"
              onClick={onClose}
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="modal-body diag-body">
          {healthState.error && (
            <div className="alert-error" role="alert">
              {healthState.error}
            </div>
          )}

          <div className="diag-grid">
            <div className="diag-section-card">
              <h4>
                <Database size={16} />
                <span>Model Status &amp; Active Weights</span>
              </h4>
              <div className="kv-list">
                <div className="kv-row">
                  <span>Active Weights</span>
                  <strong className="mono-text">{modelStatus.active_weights || 'N/A'}</strong>
                </div>
                <div className="kv-row">
                  <span>Weights Path</span>
                  <strong className="mono-text truncate-path">{modelStatus.weights_path || 'N/A'}</strong>
                </div>
                <div className="kv-row">
                  <span>Gemini Vision Model</span>
                  <strong className="mono-text">{geminiStatus.vision_model || 'gemini-3.5-flash-lite'}</strong>
                </div>
              </div>

              <div className="flags-grid">
                <StatusFlag label="YOLOv8n Loaded" ok={Boolean(modelStatus.yolov8n_loaded)} />
                <StatusFlag label="Fine-Tuned .pt" ok={Boolean(modelStatus.fine_tuned_pt_available)} />
                <StatusFlag label="Fine-Tuned .onnx" ok={Boolean(modelStatus.fine_tuned_onnx_available)} />
                <StatusFlag label="Ultralytics" ok={Boolean(modelStatus.ultralytics_installed)} />
                <StatusFlag label="ORB Baseline" ok={Boolean(modelStatus.orb_ready)} />
                <StatusFlag label="SIFT Baseline" ok={Boolean(modelStatus.sift_ready)} />
                <StatusFlag label="Gemini API Key" ok={Boolean(geminiStatus.api_key_configured)} />
              </div>
            </div>

            <div className="diag-section-card">
              <h4>
                <Cpu size={16} />
                <span>Hardware &amp; Runtime</span>
              </h4>
              <div className="kv-list">
                <div className="kv-row">
                  <span>OS / Release</span>
                  <strong>{hardware.os || '--'} {hardware.os_release || ''}</strong>
                </div>
                <div className="kv-row">
                  <span>Architecture</span>
                  <strong>{hardware.machine || '--'}</strong>
                </div>
                <div className="kv-row">
                  <span>Processor</span>
                  <strong>{hardware.processor || '--'}</strong>
                </div>
                <div className="kv-row">
                  <span>CPU Threads</span>
                  <strong>{hardware.cpu_threads || '--'}</strong>
                </div>
                <div className="kv-row">
                  <span>OpenCV Version</span>
                  <strong>{hardware.opencv_version || '--'}</strong>
                </div>
              </div>

              <div className="modes-box">
                <span className="section-label">Supported Detector Modes</span>
                <div className="pill-wrap">
                  {modes.map(function renderModePill(m) {
                    return (
                      <span key={m} className="staple-pill mono-text">
                        {m}
                      </span>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>

          <div className="diag-section-card">
            <h4>
              <Server size={16} />
              <span>Supported Unified Classes ({classes.length})</span>
            </h4>
            <div className="table-scroll">
              <table className="det-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Class ID</th>
                    <th>Display Name</th>
                    <th>Storage Zone</th>
                    <th>Est. kcal / unit</th>
                  </tr>
                </thead>
                <tbody>
                  {classes.map(function renderClassRow(cls) {
                    return (
                      <tr key={cls.class_id}>
                        <td>{cls.id}</td>
                        <td className="mono-text">{cls.class_id}</td>
                        <td><strong>{cls.display_name}</strong></td>
                        <td>{cls.storage_zone}</td>
                        <td>{getCaloriesPerUnit(cls.class_id)} kcal</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="diag-section-card">
            <div className="panel-heading">
              <h4>Session Telemetry Logs ({filteredLogs.length})</h4>
              <div className="action-row">
                {['ALL', 'INFO', 'WARN', 'ERROR', 'TELEMETRY'].map(function renderFilter(lvl) {
                  return (
                    <button
                      key={lvl}
                      type="button"
                      className={`kf-btn ${logFilter === lvl ? 'kf-btn-active' : ''}`}
                      onClick={function pickLogFilter() { setLogFilter(lvl) }}
                    >
                      {lvl}
                    </button>
                  )
                })}
                <button
                  type="button"
                  className="btn-icon-danger"
                  onClick={onClearLogs}
                  title="Clear logs"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
            <div className="logs-scroll">
              {filteredLogs.length === 0 ? (
                <p className="helper-text">No telemetry records for this filter yet.</p>
              ) : (
                filteredLogs.map(function renderLog(entry) {
                  return (
                    <div key={entry.id} className="log-entry">
                      <span className={`log-level log-${String(entry.level).toLowerCase()}`}>{entry.level}</span>
                      <span className="log-tag">[{entry.tag}]</span>
                      <span className="log-msg">{entry.message}</span>
                      <span className="log-time">{entry.time}</span>
                    </div>
                  )
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
