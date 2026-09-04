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

  // Face select / exclude styles
  const style = document.createElement('style');
  style.textContent = `
    .char-face-wrap { position: relative; flex-shrink: 0; }
    .char-face-wrap:hover .char-face-x { opacity: 1 !important; }
    .char-face-img { width: 72px; height: 72px; object-fit: cover; border-radius: 8px; border: 2px solid var(--border); cursor: pointer; display:block; }
    .char-face-img.selected { border-color: #22c55e !important; box-shadow: 0 0 0 2px rgba(34,197,94,0.35); }
    .char-face-x { position:absolute; top:-6px; right:-6px; width:20px; height:20px; background:#ef4444; color:#fff; border-radius:50%; font-size:11px; display:flex; align-items:center; justify-content:center; cursor:pointer; opacity:0; transition:opacity .15s; z-index:2; border:none; }
    .char-face-check { position:absolute; left:4px; top:4px; width:16px; height:16px; border-radius:3px; border:1px solid rgba(255,255,255,.8); background:rgba(0,0,0,.35); pointer-events:none; }
    .char-face-img.selected + .char-face-x + .char-face-check,
    .char-face-wrap .char-face-img.selected ~ .char-face-check { background:#22c55e; border-color:#22c55e; }
    .char-face-meta { font-size:10px; color:var(--text-muted); text-align:center; margin-top:2px; max-width:72px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  `;
  document.head.appendChild(style);
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
        // Show more faces so cleanup is possible (was hard-capped at 6)
        const obs = await api(`/characters/${c.id}/observations?limit=48`);
        faces = (obs || []).filter(o => o.face_crop_path && o.id && !o.excluded);
      } catch (e) {}

      const statusBadge = c.status === 'manual' ? '<span style="font-size:10px;background:rgba(34,197,94,0.15);color:#22c55e;padding:2px 8px;border-radius:10px;margin-left:8px">MANUAL</span>' :
                          c.status === 'unknown' ? '<span style="font-size:10px;background:rgba(255,255,255,0.1);color:#888;padding:2px 8px;border-radius:10px;margin-left:8px">UNKNOWN</span>' :
                          '';

      const facesHtml = faces.length ? `<div class="char-face-grid" style="display:flex;gap:8px;overflow-x:auto;padding:6px 0;flex-wrap:wrap">${faces.map(o => {
        const url = '/media/' + o.face_crop_path.replace(/^.*?character-identity-board-data\/projects\//, '');
        const conf = (o.identity_confidence != null) ? Number(o.identity_confidence).toFixed(2) : '-';
        const qs = (o.quality_score != null) ? Number(o.quality_score).toFixed(2) : '-';
        return `<div class="char-face-wrap" data-obs="${o.id}" data-char="${c.id}" data-tracklet="${o.tracklet_id}">
          <img src="${url}" class="char-face-img" data-obs="${o.id}" data-char="${c.id}" data-tracklet="${o.tracklet_id}"
            title="Shot #${o.shot_number} · q=${qs} · conf=${conf} — click to select"
            onclick="toggleOneFace(event, this)" onerror="this.parentElement.style.display='none'">
          <button type="button" class="char-face-x" title="排除这张脸" onclick="event.stopPropagation();excludeObservation(${o.id}, this.parentElement.querySelector('img'))">✕</button>
          <div class="char-face-check"></div>
          <div class="char-face-meta">#${o.shot_number}</div>
        </div>`;
      }).join('')}</div>` : '<div style="font-size:12px;color:var(--text-muted)">No face crops</div>';

      const pendingHtml = c.pending_review > 0 ? `<div style="font-size:12px;color:var(--yellow)">⚠ ${c.pending_review} pending review</div>` : '';

      const batchBar = `<div style="display:flex;gap:6px;margin-top:8px;align-items:center;flex-wrap:wrap">
        <button type="button" onclick="toggleFaceSelect(${c.id})" style="background:var(--card);color:var(--foreground);border:1px solid var(--border);padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px">☐ 全选/取消</button>
        <button type="button" onclick="batchExcludeFaces(${c.id})" style="background:#ef4444;color:white;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px">✗ 排除选中</button>
        <button type="button" onclick="batchUnknownFaces(${c.id})" style="background:#f59e0b;color:white;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px">? 移到 Unknown</button>
        <span style="font-size:11px;color:var(--text-muted)">显示 ${faces.length} 张 · 点图选择，点 ✕ 单张排除</span>
      </div>`;

      html += `
        <div class="character-card" data-char-card="${c.id}" style="flex-direction:column;align-items:stretch;gap:12px">
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
          ${batchBar}
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

// ═══════════════════════════════════════════════════════
// REVIEW QUEUE — Batch selection + optimistic updates
// ═══════════════════════════════════════════════════════

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
    renderReviewQueue(items);
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-text">Error: ${esc(e.message)}</div></div>`;
  }
}

function renderReviewQueue(items) {
  const container = document.getElementById('review-container');

  const batchBar = `
    <div id="review-batch-bar" style="display:flex;gap:10px;margin-bottom:16px;align-items:center;flex-wrap:wrap">
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:13px;color:var(--text-secondary)">
        <input type="checkbox" id="review-select-all" onchange="toggleSelectAll(this)" style="width:18px;height:18px;cursor:pointer">
        全选 (<span id="review-selected-count">0</span>/${items.length})
      </label>
      <button class="btn btn-sm" style="background:#ef4444;color:white" onclick="batchNotAFace()">✗ Not a face (批量)</button>
      <button class="btn btn-success btn-sm" onclick="batchConfirm()">✓ Confirm All High-Conf (≥0.90)</button>
      <button class="btn btn-ghost btn-sm" onclick="batchExcludeLow()">✗ Exclude Low-Conf (&lt;0.70)</button>
      <span style="font-size:13px;color:var(--text-muted);margin-left:auto">${items.length} items pending</span>
    </div>`;

  container.innerHTML = batchBar + items.map(r => {
    const cropUrl = r.face_crop_path
      ? `/media/${r.face_crop_path.replace(/^.*?character-identity-board-data\/projects\//, '')}`
      : '';
    const reasons = (r.reasons || []).map(reason =>
      `<span class="review-reason">${esc(reason)}</span>`
    ).join(' ');
    return `
      <div class="review-card ${r.review_status}" id="review-card-${r.tracklet_id}">
        <input type="checkbox" class="review-checkbox" data-tracklet="${r.tracklet_id}" 
               onchange="updateSelectedCount()" style="width:18px;height:18px;cursor:pointer;flex-shrink:0">
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
}

// ─── Batch Selection ────────────────────────────────────
function toggleSelectAll(checkbox) {
  const all = document.querySelectorAll('.review-checkbox');
  all.forEach(cb => { cb.checked = checkbox.checked; });
  updateSelectedCount();
}

function updateSelectedCount() {
  const checked = document.querySelectorAll('.review-checkbox:checked').length;
  const el = document.getElementById('review-selected-count');
  if (el) el.textContent = checked;
}

function getSelectedTracklets() {
  return [...document.querySelectorAll('.review-checkbox:checked')].map(cb => cb.dataset.tracklet);
}

// Optimistic removal: fade out and remove cards without full page reload
function removeReviewCards(trackletIds) {
  trackletIds.forEach(id => {
    const card = document.getElementById('review-card-' + id);
    if (card) {
      card.style.transition = 'opacity 0.2s, transform 0.2s';
      card.style.opacity = '0';
      card.style.transform = 'translateX(30px)';
      setTimeout(() => card.remove(), 200);
    }
  });
  // Update count
  setTimeout(() => {
    updateSelectedCount();
    const remaining = document.querySelectorAll('.review-card').length;
    const batchBar = document.getElementById('review-batch-bar');
    if (batchBar) {
      const countSpan = batchBar.querySelector('span:last-child');
      if (countSpan) countSpan.textContent = remaining + ' items pending';
    }
    if (remaining === 0) {
      document.getElementById('review-container').innerHTML = 
        `<div class="empty-state"><div class="empty-state-text">Review queue is empty 🎉</div></div>`;
    }
  }, 250);
}

// ─── Batch Actions ──────────────────────────────────────

// Batch: mark selected as "Not a face"
async function batchNotAFace() {
  const ids = getSelectedTracklets();
  if (!ids.length) { toast('请先选择要排除的项', 'error'); return; }
  
  toast(`正在排除 ${ids.length} 项...`, 'info');
  
  // Optimistic: remove from UI immediately
  removeReviewCards(ids);
  
  // Send API requests in parallel
  const results = await Promise.allSettled(ids.map(id =>
    api(`/tracklets/${id}/assignment`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: 'confirmed', note: 'Excluded: not a real face (batch)' })
    })
  ));
  
  const ok = results.filter(r => r.status === 'fulfilled').length;
  const fail = results.filter(r => r.status === 'rejected').length;
  
  if (fail) {
    toast(`排除 ${ok} 项成功，${fail} 项失败`, fail === ids.length ? 'error' : 'info');
    if (ok === 0) {
      // All failed, reload to restore state
      if (currentProject) loadReviewQueue(currentProject.id);
    }
  } else {
    toast(`✓ 已排除 ${ok} 项 (Not a face)`, 'success');
  }
  
  // Refresh characters count
  if (currentProject) loadCharacters(currentProject.id);
}

// Confirm high-confidence (existing, improved with optimistic update)
async function batchConfirm() {
  if (!currentProject) return;
  try {
    const items = await api(`/projects/${currentProject.id}/review-queue`);
    const highConf = items.filter(r => r.identity_confidence >= 0.90);
    if (!highConf.length) { toast('没有高置信度项', 'info'); return; }
    
    toast(`正在确认 ${highConf.length} 项...`, 'info');
    
    // Optimistic
    removeReviewCards(highConf.map(r => String(r.tracklet_id)));
    
    const results = await Promise.allSettled(highConf.map(r =>
      api(`/tracklets/${r.tracklet_id}/assignment`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_status: 'confirmed' })
      })
    ));
    
    const ok = results.filter(r => r.status === 'fulfilled').length;
    toast(`✓ 已确认 ${ok} 项高置信度`, 'success');
    if (currentProject) loadCharacters(currentProject.id);
  } catch (e) {
    toast('Batch confirm failed: ' + e.message, 'error');
  }
}

// Exclude low-confidence (existing, improved with optimistic update)
async function batchExcludeLow() {
  if (!currentProject) return;
  try {
    const items = await api(`/projects/${currentProject.id}/review-queue`);
    const lowConf = items.filter(r => r.identity_confidence < 0.70);
    if (!lowConf.length) { toast('没有低置信度项', 'info'); return; }
    
    toast(`正在排除 ${lowConf.length} 项...`, 'info');
    
    // Optimistic
    removeReviewCards(lowConf.map(r => String(r.tracklet_id)));
    
    const results = await Promise.allSettled(lowConf.map(r =>
      api(`/tracklets/${r.tracklet_id}/assignment`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character_id: 0, note: 'Excluded: low confidence' })
      })
    ));
    
    const ok = results.filter(r => r.status === 'fulfilled').length;
    toast(`✓ 已排除 ${ok} 项低置信度`, 'success');
    if (currentProject) loadCharacters(currentProject.id);
  } catch (e) {
    toast('Batch exclude failed: ' + e.message, 'error');
  }
}

// ─── Single Actions (optimistic, no full reload) ────────

async function confirmReview(trackletId) {
  try {
    await api(`/tracklets/${trackletId}/assignment`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review_status: 'confirmed' })
    });
    toast('Review confirmed', 'success');
    removeReviewCards([String(trackletId)]);
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
    removeReviewCards([String(trackletId)]);
    if (currentProject) loadCharacters(currentProject.id);
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
    removeReviewCards([String(trackletId)]);
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

// ─── Character face select / exclude ───────────────────
function toggleOneFace(ev, imgEl) {
  if (ev) ev.preventDefault();
  imgEl.classList.toggle('selected');
}

function getSelectedFaceImgs(charId) {
  return [...document.querySelectorAll(`.char-face-img[data-char="${charId}"].selected`)];
}

function removeFaceWrapByObs(obsId) {
  document.querySelectorAll(`.char-face-wrap[data-obs="${obsId}"]`).forEach(wrap => {
    wrap.style.transition = 'opacity 0.2s, transform 0.2s';
    wrap.style.opacity = '0';
    wrap.style.transform = 'scale(0.85)';
    setTimeout(() => wrap.remove(), 200);
  });
}

async function excludeObservation(obsId, imgEl) {
  if (!obsId || obsId === 'undefined' || Number.isNaN(Number(obsId))) {
    toast('无法排除：缺少 observation id', 'error');
    return;
  }
  const wrap = imgEl ? (imgEl.closest('.char-face-wrap') || imgEl.parentElement) : null;
  try {
    if (wrap) {
      wrap.style.transition = 'opacity 0.2s, transform 0.2s';
      wrap.style.opacity = '0.4';
    }
    await api(`/observations/${obsId}/exclude`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ excluded: true, reason: 'excluded by user (click)' })
    });
    toast('✓ 已排除这张脸', 'success');
    removeFaceWrapByObs(obsId);
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
    if (wrap) wrap.style.opacity = '1';
  }
}

async function batchExcludeFaces(charId) {
  const checked = getSelectedFaceImgs(charId);
  if (!checked.length) { toast('请先点选要排除的脸（绿色边框）', 'error'); return; }

  const items = checked.map(img => ({
    obsId: img.dataset.obs,
    trackletId: img.dataset.tracklet,
  })).filter(x => x.obsId);

  toast(`正在排除 ${items.length} 张脸...`, 'info');

  const results = await Promise.allSettled(items.map(it =>
    api(`/observations/${it.obsId}/exclude`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ excluded: true, reason: 'batch excluded by user' })
    })
  ));

  let ok = 0;
  results.forEach((r, idx) => {
    if (r.status === 'fulfilled') {
      ok += 1;
      removeFaceWrapByObs(items[idx].obsId);
    }
  });
  toast(`✓ 已排除 ${ok}/${items.length} 张脸`, ok ? 'success' : 'error');
}

async function batchUnknownFaces(charId) {
  const checked = getSelectedFaceImgs(charId);
  if (!checked.length) { toast('请先点选要移动的脸', 'error'); return; }

  const trackletIds = [...new Set(checked.map(img => img.dataset.tracklet).filter(Boolean))];
  toast(`正在移到 Unknown：${trackletIds.length} 个 tracklet...`, 'info');

  const results = await Promise.allSettled(trackletIds.map(tid =>
    api(`/tracklets/${tid}/assignment`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character_id: 0, note: 'Moved to Unknown by batch UI' })
    })
  ));

  const ok = results.filter(r => r.status === 'fulfilled').length;
  toast(`✓ 已移动 ${ok}/${trackletIds.length}`, ok ? 'success' : 'error');

  // Remove selected wraps locally without full reload
  checked.forEach(img => removeFaceWrapByObs(img.dataset.obs));
}

function toggleFaceSelect(charId) {
  const imgs = [...document.querySelectorAll(`.char-face-img[data-char="${charId}"]`)];
  if (!imgs.length) return;
  const allSelected = imgs.every(img => img.classList.contains('selected'));
  imgs.forEach(img => {
    img.classList.toggle('selected', !allSelected);
  });
}

async function confirmWithChar(trackletId) {
  const select = document.getElementById(`char-select-${trackletId}`);
  const charId = select ? select.value : '';
  
  try {
    if (charId) {
      await api(`/tracklets/${trackletId}/assignment`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ character_id: parseInt(charId), review_status: 'confirmed' })
      });
      toast('Assigned and confirmed', 'success');
    } else {
      await confirmReview(trackletId);
      return;
    }
    removeReviewCards([String(trackletId)]);
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
    removeReviewCards([String(trackletId)]);
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
  }
}

// ─── Populate dropdowns ─────────────────────────────────
async function populateCharDropdowns() {
  if (!currentProject) return;
  try {
    const chars = await api(`/projects/${currentProject.id}/characters`);
    const selects = document.querySelectorAll('.review-char-select');
    selects.forEach(sel => {
      const currentVal = sel.value;
      sel.innerHTML = '<option value="">-- 选择人物 --</option>';
      chars.forEach(c => {
        if (c.character_code === 'UNKNOWN') return;
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

// ─── Processing Monitor ─────────────────────────────────
async function refreshProcessing() {
  const container = document.getElementById('processing-container');
  try {
    const projects = await api('/projects');
    const processing = projects.filter(p => p.status === 'processing');

    if (!processing.length) {
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
async function recluster() {
  if (!currentProject) return;
  try {
    await api(`/projects/${currentProject.id}/recluster`, { method: 'POST' });
    toast('Recluster started', 'info');
  } catch (e) {
    toast('Failed: ' + e.message, 'error');
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
