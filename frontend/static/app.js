/* === Character Identity Board — Dashboard JS === */

const API = '';  // same origin
let currentProject = null;
let refreshTimer = null;

// ─── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
  // Auto-refresh processing view every 8s
  setInterval(() => {
    const procView = document.getElementById('view-processing');
    if (procView && procView.classList.contains('active')) refreshProcessing();
  }, 8000);
});

// ─── API Helpers ────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  if (res.status === 204) return null;
  return res.json();
}

function toast(msg, type = 'info') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ─── Views ──────────────────────────────────────────────
function switchView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const view = document.getElementById('view-' + name);
  if (view) view.classList.add('active');
  const nav = document.querySelector(`.nav-item[data-view="${name}"]`);
  if (nav) nav.classList.add('active');

  if (name === 'dashboard') loadDashboard();
  if (name === 'projects') loadDashboard();
  if (name === 'processing') refreshProcessing();
}

function switchTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  if (btn) btn.classList.add('active');
  const tab = document.getElementById('tab-' + name);
  if (tab) tab.classList.add('active');
}

// ─── Dashboard ──────────────────────────────────────────
async function loadDashboard() {
  try {
    const projects = await api('/projects');
    let totalShots = 0, totalChars = 0, processing = 0;
    projects.forEach(p => {
      totalShots += p.shot_count || 0;
      totalChars += p.character_count || 0;
      if (p.status === 'processing') processing++;
    });

    document.getElementById('stat-projects').textContent = projects.length;
    document.getElementById('stat-shots').textContent = totalShots.toLocaleString();
    document.getElementById('stat-characters').textContent = totalChars;
    document.getElementById('stat-processing').textContent = processing;

    const grid = document.getElementById('project-grid');
    if (!projects.length) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column:1/-1">
          <div class="empty-state-icon">◉</div>
          <div class="empty-state-text">No projects yet. Create one to get started.</div>
        </div>`;
      return;
    }

    grid.innerHTML = projects.map(p => {
      const statusClass = p.status === 'processing' ? 'status-processing' :
                          p.status === 'created' ? 'status-created' : 'status-has_video';
      const statusLabel = p.status === 'has_video' ? 'READY' : p.status.toUpperCase();
      return `
        <div class="project-card" onclick="openProject(${p.id})">
          <div class="project-card-header">
            <div class="project-name">${esc(p.name)}</div>
            <span class="project-status ${statusClass}">${statusLabel}</span>
          </div>
          <div class="project-stats">
            <div class="project-stat">
              <div class="project-stat-value">${p.video_count}</div>
              <div class="project-stat-label">Videos</div>
            </div>
            <div class="project-stat">
              <div class="project-stat-value">${(p.shot_count || 0).toLocaleString()}</div>
              <div class="project-stat-label">Shots</div>
            </div>
            <div class="project-stat">
              <div class="project-stat-value">${p.character_count || 0}</div>
              <div class="project-stat-label">Characters</div>
            </div>
          </div>
          <div class="project-date">${formatDate(p.created_at)}</div>
        </div>`;
    }).join('');
  } catch (e) {
    toast('Failed to load dashboard: ' + e.message, 'error');
  }
}

// ─── Project Detail ─────────────────────────────────────
async function openProject(id) {
  try {
    currentProject = await api(`/projects/${id}`);
    document.getElementById('detail-title').textContent = currentProject.name;
    document.getElementById('detail-shots').textContent = (currentProject.shot_count || 0).toLocaleString();
    document.getElementById('detail-characters').textContent = currentProject.character_count || 0;
    document.getElementById('detail-videos').textContent = currentProject.video_count || 0;

    // Show detail view
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById('view-detail').classList.add('active');

    // Load all data
    loadShots(id);
    loadCharacters(id);
    loadReviewQueue(id);
  } catch (e) {
    toast('Failed to load project: ' + e.message, 'error');
  }
}

async function loadShots(projectId) {
  const container = document.getElementById('shots-container');
  container.innerHTML = '<div class="loading">Loading shots</div>';
  try {
    const shots = await api(`/projects/${projectId}/shots`);
    if (!shots.length) {
      container.innerHTML = `<div class="empty-state"><div class="empty-state-text">No shots detected yet</div></div>`;
      return;
    }
    container.innerHTML = shots.map(s => {
      const thumbUrl = s.representative_frame
        ? `/media/${s.representative_frame.replace(/^.*?character-identity-board-data\/projects\//, '')}`
        : '';
      return `
        <div class="shot-card">
          ${thumbUrl ? `<img class="shot-thumb" src="${thumbUrl}" alt="Shot ${s.shot_number}" loading="lazy" onerror="this.style.display='none'">` :
            `<div class="shot-thumb" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:12px">No Frame</div>`}
          <div class="shot-info">
            <div class="shot-number">Shot #${s.shot_number}</div>
            <div class="shot-timecode">${s.timecode_start} → ${s.timecode_end}</div>
          </div>
        </div>`;
    }).join('');
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error loading shots: ${esc(e.message)}</div></div>`;
  }
}

async function loadCharacters(projectId) {
  const container = document.getElementById('characters-container');
  container.innerHTML = '<div class="loading">Loading characters</div>';
  try {
    const chars = await api(`/projects/${projectId}/characters`);
    const clusters = chars.filter(c => c.character_code !== 'UNKNOWN');
    
    if (!clusters.length) {
      container.innerHTML = `<div class="empty-state"><div class="empty-state-text">No characters detected yet</div></div>`;
      return;
    }

    let html = '';
    for (const c of clusters) {
      let faces = [];
      try {
        const obs = await api(`/characters/${c.id}/observations?limit=6`);
        faces = obs.filter(o => o.face_crop_path).slice(0, 6);
      } catch(e) {}

      const statusBadge = c.status === 'manual' ? '<span style="font-size:10px;background:rgba(34,197,94,0.15);color:#22c55e;padding:2px 8px;border-radius:10px;margin-left:8px">MANUAL</span>' :
                          c.status === 'unknown' ? '<span style="font-size:10px;background:rgba(255,255,255,0.1);color:#888;padding:2px 8px;border-radius:10px;margin-left:8px">UNKNOWN</span>' :
                          '';

      const facesHtml = faces.length ? `<div style="display:flex;gap:6px;overflow-x:auto;padding:4px 0">${faces.map(o => {
        const url = '/media/' + o.face_crop_path.replace(/^.*?character-identity-board-data\/projects\//, '');
        return `<img src="${url}" style="width:56px;height:56px;object-fit:cover;border-radius:8px;border:2px solid var(--border);flex-shrink:0" title="Shot #${o.shot_number} q=${(o.quality_score||0).toFixed(2)}" onerror="this.style.display='none'">`;
      }).join('')}</div>` : '';

      const pendingHtml = c.pending_review > 0 ? `<div style="font-size:12px;color:var(--yellow)">⚠ ${c.pending_review} pending review</div>` : '';

      html += `
        <div class="character-card" style="flex-direction:column;align-items:stretch;gap:12px">
          <div style="display:flex;align-items:center;gap:12px">
            <div class="character-avatar" style="width:48px;height:48px;font-size:18px">${esc(c.display_name.charAt(0))}</div>
            <div style="flex:1">
              <div style="display:flex;align-items:center;gap:8px">
                <input type="text" id="cname${c.id}" value="${esc(c.display_name)}" 
                  style="background:transparent;border:none;border-bottom:2px solid var(--border);color:var(--foreground);font-size:16px;font-weight:600;width:180px;padding:2px 0;outline:none" />
                ${statusBadge}
              </div>
              <div style="font-size:12px;color:var(--text-muted);margin-top:2px">${esc(c.character_code)} · ${c.tracklet_count} tracklets · ${c.shot_count} shots · confidence: ${(c.avg_confidence || 0).toFixed(2)}</div>
            </div>
            <div style="display:flex;gap:8px">
              <button onclick="renameChar(${c.id})" style="background:#22c55e;color:white;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500">✓ 重命名</button>
              <button onclick="deleteChar(${c.id})" style="background:#ef4444;color:white;border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500">✗ 删除</button>
            </div>
          </div>
          ${facesHtml}
          ${pendingHtml}
        </div>`;
    }
    
    container.innerHTML = html;
    document.getElementById('detail-review').textContent = chars.reduce((sum, c) => sum + (c.pending_review || 0), 0);
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${esc(e.message)}</div></div>`;
  }
}


async function renameChar(id) {
  const input = document.getElementById('cname' + id);
  const name = input.value.trim();
  if (!name) return;
  try {
    await api('/characters/' + id, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ display_name: name })
    });
    toast('已重命名为: ' + name, 'success');
    loadCharacters(currentProject.id);
  } catch (e) {
    toast('失败: ' + e.message, 'error');
  }
}

async function deleteChar(id) {
  if (!confirm('确定删除这个 Cluster？')) return;
  try {
    await api('/characters/' + id, { method: 'DELETE' });
    toast('已删除', 'success');
    loadCharacters(currentProject.id);
  } catch (e) {
    toast('失败: ' + e.message, 'error');
  }
}

let cachedChars = null;
async function getChars() {
  if (!cachedChars) {
    cachedChars = await api('/projects/' + currentProject.id + '/characters');
  }
  return cachedChars;
}

async function loadReviewQueue(projectId) {
  cachedChars = null; // refresh
  const container = document.getElementById('review-container');
  container.innerHTML = '<div class="loading">Loading review queue</div>';
  try {
    const items = await api(`/projects/${projectId}/review-queue`);
    if (!items.length) {
      container.innerHTML = `<div class="empty-state"><div class="empty-state-text">Review queue is empty 🎉</div></div>`;
      return;
    }
    // Add batch action bar
    const batchBar = `
      <div style="display:flex;gap:10px;margin-bottom:16px;align-items:center;flex-wrap:wrap">
        <span style="font-size:13px;color:var(--text-secondary)">${items.length} items pending review</span>
        <button class="btn btn-success btn-sm" onclick="batchConfirm()">✓ Confirm All High-Conf (≥0.90)</button>
        <button class="btn btn-ghost btn-sm" onclick="batchExcludeLow()">✗ Exclude Low-Conf (&lt;0.70)</button>
      </div>`;

    container.innerHTML = batchBar + items.map(r => {
      const cropUrl = r.face_crop_path
        ? `/media/${r.face_crop_path.replace(/^.*?character-identity-board-data\/projects\//, '')}`
        : '';
      const reasons = (r.reasons || []).map(reason =>
        `<span class="review-reason">${esc(reason)}</span>`
      ).join(' ');
      return `
        <div class="review-card ${r.review_status}">
          ${cropUrl ? `<img class="review-crop" src="${cropUrl}" alt="face" loading="lazy" onerror="this.style.display='none'">` :
            `<div class="review-crop" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:11px">N/A</div>`}
          <div class="review-info">
            <div class="review-shot">Shot #${r.shot_number} · ${r.timecode_start}</div>
            <div class="review-character">${esc(r.character_name)} <span style="color:var(--text-muted);font-weight:400">(${esc(r.character_code)})</span></div>
            <div class="review-details">
              <span>confidence: <strong>${(r.identity_confidence || 0).toFixed(3)}</strong></span>
              <span>quality: <strong>${(r.quality_score || 0).toFixed(3)}</strong></span>
              <span>source: ${esc(r.assignment_source)}</span>
              ${reasons}
            </div>
          </div>
          <div class="review-actions">
            ${r.review_status === 'pending' ? `
              <select class="review-char-select" id="char-select-${r.tracklet_id}" style="
                background:var(--card);color:var(--foreground);border:1px solid var(--border);
                border-radius:4px;padding:4px 6px;font-size:12px;max-width:140px;
              ">
                <option value="">-- 选择人物 --</option>
              </select>
              <button class="btn btn-success btn-sm" onclick="confirmWithChar(${r.tracklet_id})">✓ 确认</button>
              <button class="btn btn-ghost btn-sm" onclick="markUnknown(${r.tracklet_id})">Unknown</button>
              <button class="btn btn-sm" style="background:var(--red);color:white" onclick="excludeNotFace(${r.tracklet_id})">✗ Not a face</button>
            ` : `<span style="font-size:11px;color:var(--green)">✓ Confirmed</span>`}
          </div>
        </div>`;
    }).join('');
    // Populate character dropdowns after rendering
    populateCharDropdowns();
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${esc(e.message)}</div></div>`;
  }
}

// ─── Processing Monitor ─────────────────────────────────
async function refreshProcessing() {
  const container = document.getElementById('processing-container');
  try {
    const projects = await api('/projects');
    const processing = projects.filter(p => p.status === 'processing');

    if (!processing.length) {
      // Also check all projects for videos being processed
      let allVideos = [];
      for (const p of projects) {
        try {
          const videos = await api(`/projects/${p.id}/videos`);
          videos.forEach(v => { v._project = p; allVideos.push(v); });
        } catch(e) {}
      }
      const activeVideos = allVideos.filter(v => v.processing_status === 'processing');

      if (!activeVideos.length) {
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-state-icon">↻</div>
            <div class="empty-state-text">No active processing jobs</div>
          </div>`;
        return;
      }

      container.innerHTML = activeVideos.map(v => renderProcessingVideo(v, v._project)).join('');
      return;
    }

    // Load videos for processing projects
    let html = '';
    for (const p of processing) {
      const videos = await api(`/projects/${p.id}/videos`);
      for (const v of videos) {
        html += renderProcessingVideo(v, p);
      }
    }
    container.innerHTML = html || '<div class="empty-state"><div class="empty-state-text">No active processing</div></div>';
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${esc(e.message)}</div></div>`;
  }
}

function renderProcessingVideo(video, project) {
  // Estimate progress from pipeline stage
  const stages = ['uploaded', 'probed', 'shots_detected', 'tracklets_created',
                  'faces_embedded', 'clustered', 'thumbnails_generated', 'completed'];
  const stageIndex = stages.indexOf(video.pipeline_stage);
  const stageProgress = stageIndex >= 0 ? Math.round((stageIndex / stages.length) * 100) : 0;

  const duration = video.duration_s ? formatDuration(video.duration_s) : '-';
  const resolution = video.width && video.height ? `${video.width}×${video.height}` : '-';
  const fps = video.fps || '-';
  const frames = video.frame_count ? video.frame_count.toLocaleString() : '-';

  return `
    <div class="processing-card">
      <div class="processing-header">
        <div>
          <div class="processing-title">${esc(project.name)}</div>
          <div style="font-size:12px;color:var(--text-muted);margin-top:2px">${esc(video.filename)}</div>
        </div>
        <span class="processing-stage">${esc(video.pipeline_stage || 'unknown')}</span>
      </div>
      <div class="progress-bar-container">
        <div class="progress-bar" style="width:${stageProgress}%"></div>
      </div>
      <div class="processing-stats">
        <span class="processing-stat">Duration: <strong>${duration}</strong></span>
        <span class="processing-stat">Resolution: <strong>${resolution}</strong></span>
        <span class="processing-stat">FPS: <strong>${fps}</strong></span>
        <span class="processing-stat">Frames: <strong>${frames}</strong></span>
        <span class="processing-stat">Progress: <strong>${stageProgress}%</strong></span>
      </div>
    </div>`;
}

// ─── Actions ────────────────────────────────────────────
async function confirmReview(trackletId) {
  try {
    await api(`/tracklets/${trackletId}/assignment`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: 'confirmed' })
    });
    toast('Review confirmed', 'success');
    if (currentProject) loadReviewQueue(currentProject.id);
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

async function markUnknown(trackletId) {
  try {
    await api(`/tracklets/${trackletId}/assignment`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character_id: 0, note: 'Marked unknown by user' })
    });
    toast('Marked as Unknown', 'info');
    if (currentProject) {
      loadReviewQueue(currentProject.id);
      loadCharacters(currentProject.id);
    }
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

async function recluster() {
  if (!currentProject) return;
  try {
    await api(`/projects/${currentProject.id}/recluster`, { method: 'POST' });
    toast('Recluster started', 'info');
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

async function excludeNotFace(trackletId) {
  try {
    await api(`/tracklets/${trackletId}/assignment`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: 'confirmed', note: 'Excluded: not a real face' })
    });
    toast('Excluded as non-face', 'success');
    if (currentProject) loadReviewQueue(currentProject.id);
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

async function reassignTracklet(trackletId, newCharacterId) {
  try {
    await api(`/tracklets/${trackletId}/assignment`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character_id: newCharacterId, review_status: 'confirmed' })
    });
    toast('Reassigned successfully', 'success');
    if (currentProject) loadReviewQueue(currentProject.id);
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

// Confirm review with character assignment from dropdown
async function confirmWithChar(trackletId) {
  const select = document.getElementById(`char-select-${trackletId}`);
  const charId = select ? select.value : '';
  
  try {
    if (charId) {
      // Assign to specific character
      await api(`/tracklets/${trackletId}/assignment`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character_id: parseInt(charId), review_status: 'confirmed' })
      });
      toast('Assigned and confirmed', 'success');
    } else {
      // No character selected, just confirm current assignment
      await confirmReview(trackletId);
      return;
    }
    if (currentProject) loadReviewQueue(currentProject.id);
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

// Populate character dropdowns in review queue
async function populateCharDropdowns() {
  if (!currentProject) return;
  try {
    const chars = await api(`/projects/${currentProject.id}/characters`);
    const selects = document.querySelectorAll('.review-char-select');
    selects.forEach(sel => {
      const currentVal = sel.value;
      sel.innerHTML = '<option value="">-- 选择人物 --</option>';
      chars.forEach(c => {
        if (c.character_code === 'UNKNOWN') return; // Skip Unknown
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = `${c.display_name} (${c.character_code})`;
        sel.appendChild(opt);
      });
      sel.value = currentVal;
    });
  } catch (e) {
    console.error('Failed to load characters for dropdown:', e);
  }
}

async function batchConfirm() {
  if (!currentProject) return;
  try {
    const items = await api(`/projects/${currentProject.id}/review-queue`);
    const highConf = items.filter(r => r.identity_confidence >= 0.90);
    let done = 0;
    for (const r of highConf) {
      await api(`/tracklets/${r.tracklet_id}/assignment`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_status: 'confirmed' })
      });
      done++;
    }
    toast(`Confirmed ${done} high-confidence items`, 'success');
    loadReviewQueue(currentProject.id);
    loadCharacters(currentProject.id);
  } catch (e) {
    toast('Batch confirm failed: ' + e.message, 'error');
  }
}

async function batchExcludeLow() {
  if (!currentProject) return;
  try {
    const items = await api(`/projects/${currentProject.id}/review-queue`);
    const lowConf = items.filter(r => r.identity_confidence < 0.70);
    let done = 0;
    for (const r of lowConf) {
      await api(`/tracklets/${r.tracklet_id}/assignment`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character_id: 0, note: 'Excluded: low confidence' })
      });
      done++;
    }
    toast(`Excluded ${done} low-confidence items`, 'success');
    loadReviewQueue(currentProject.id);
    loadCharacters(currentProject.id);
  } catch (e) {
    toast('Batch exclude failed: ' + e.message, 'error');
  }
}

async function exportProject() {
  if (!currentProject) return;
  try {
    const result = await api(`/projects/${currentProject.id}/export`, { method: 'POST' });
    toast('Export complete!', 'success');
    if (result && result.files) {
      console.log('Exported files:', result.files);
    }
  } catch (e) {
    toast('Export failed: ' + e.message, 'error');
  }
}

// ─── Create Project ─────────────────────────────────────
function createProject() {
  document.getElementById('modal-title').textContent = 'Create New Project';
  document.getElementById('modal-body').innerHTML = `
    <div class="input-group">
      <label>Project Name</label>
      <input type="text" id="new-project-name" placeholder="My Film Project" autofocus>
    </div>
    <div class="input-group">
      <label>Video File (optional — upload later)</label>
      <input type="file" id="new-project-video" accept="video/*">
    </div>
    <div class="modal-footer">
      <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
      <button class="btn btn-primary" onclick="submitCreateProject()">Create</button>
    </div>`;
  document.getElementById('modal-overlay').classList.remove('hidden');
  setTimeout(() => document.getElementById('new-project-name').focus(), 100);
}

async function submitCreateProject() {
  const name = document.getElementById('new-project-name').value.trim();
  if (!name) { toast('Please enter a name', 'error'); return; }

  try {
    const project = await api('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    toast('Project created!', 'success');

    // Upload video if selected
    const fileInput = document.getElementById('new-project-video');
    if (fileInput.files.length > 0) {
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      toast('Uploading video...', 'info');
      await api(`/projects/${project.id}/videos`, { method: 'POST', body: formData });
      toast('Video uploaded! Starting processing...', 'info');
      await api(`/projects/${project.id}/videos/process`, { method: 'POST' });
      toast('Processing started', 'success');
    }

    closeModal();
    loadDashboard();
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

function closeModal() {
  document.getElementById('modal-overlay').classList.add('hidden');
}

// ─── Utils ──────────────────────────────────────────────
function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString('zh-CN', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function formatDuration(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}
