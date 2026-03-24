/* ═══════════════════════════════════════════════════════════════
   Micro-Coaching Simulator  |  script.js  (v2)
   Landing → Config → Chat   |  Simulating Chat + Custom Chat
═══════════════════════════════════════════════════════════════ */

'use strict';

/* ─── App state ──────────────────────────────────────────────── */
const S = {
  screen:       'landing',
  mode:         null,         // 'simulation' | 'custom'
  judgeEnabled: false,
  sessionId:    null,
  ended:        false,
  simRunning:   false,
  modelLabel:   null,       // (unused — kept for compat)
  coachLabel:   null,       // coach_llm_repo short name
  userLabel:    null,       // user_llm_repo short name
  prevConfig:   null,
};

/* ─── DOM shortcuts ──────────────────────────────────────────── */
function $(id) { return document.getElementById(id); }

const screens = {
  landing:      null,
  configSim:    null,
  configCustom: null,
  chat:         null,
};

/* ─── Init ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  screens.landing      = $('landing-screen');
  screens.configSim    = $('config-sim-screen');
  screens.configCustom = $('config-custom-screen');
  screens.chat         = $('chat-screen');

  loadGoals();
  bindEvents();
  pollServerReady();
});

/* ─── Server-ready polling ───────────────────────────────────── */
function pollServerReady() {
  const banner = document.createElement('div');
  banner.id = 'model-loading-banner';
  banner.textContent = '\u23F3  Loading AI model — first request will be ready shortly\u2026';
  document.body.appendChild(banner);

  const iv = setInterval(async () => {
    try {
      const data = await fetch('/api/status').then(r => r.json());
      if (data.ready) {
        if (data.coach_label) S.coachLabel = data.coach_label;
        if (data.user_label)  S.userLabel  = data.user_label;
        banner.textContent = '\u2713  AI model ready!';
        banner.style.background = '#2e9e6e';
        setTimeout(() => banner.remove(), 2500);
        clearInterval(iv);
      }
    } catch (_) {}
  }, 3000);
}

/* ─── Load goals into both config selects ─────────────────────── */
async function loadGoals() {
  let goals = [];
  try {
    const data = await fetch('/api/goals').then(r => r.json());
    goals = data.goals || [];
  } catch (_) {}

  ['sim-goal', 'custom-goal'].forEach(selId => {
    const sel = $(selId);
    sel.innerHTML = '';
    const ph = new Option('Select a goal\u2026', '', true, true);
    ph.disabled = true;
    sel.add(ph);
    goals.forEach(g => {
      const val   = typeof g === 'object' ? g.value : g;
      const label = typeof g === 'object' ? g.label : fmtGoal(g);
      sel.add(new Option(label, val));
    });
  });
}

function fmtGoal(key) {
  return String(key).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/* ─── Screen navigation ──────────────────────────────────────── */
function show(name) {
  Object.entries(screens).forEach(([k, el]) => {
    if (el) el.classList.toggle('active', k === name);
  });
  S.screen = name;
}

/* ─── Event bindings ─────────────────────────────────────────── */
function bindEvents() {
  /* Landing */
  $('pick-simulation').addEventListener('click', () => {
    S.mode = 'simulation';
    show('configSim');
  });
  $('pick-custom').addEventListener('click', () => {
    S.mode = 'custom';
    show('configCustom');
  });

  /* Back from config */
  $('back-from-sim-config').addEventListener('click', () => show('landing'));
  $('back-from-custom-config').addEventListener('click', () => show('landing'));

  /* Sim config form */
  $('sim-config-form').addEventListener('submit', e => {
    e.preventDefault();
    handleConfigSubmit({
      mode:         'simulation',
      goal:         $('sim-goal').value,
      mealType:     $('sim-meal-type').value,
      mealDesc:     $('sim-meal-desc').value.trim(),
      mealIngr:     $('sim-meal-ingr').value.trim(),
      judgeEnabled: $('sim-judge-toggle').checked,
      errorEl:      $('sim-form-error'),
      startBtn:     $('sim-start-btn'),
    });
  });

  /* Custom config form */
  $('custom-config-form').addEventListener('submit', e => {
    e.preventDefault();
    handleConfigSubmit({
      mode:         'custom',
      goal:         $('custom-goal').value,
      mealType:     $('custom-meal-type').value,
      mealDesc:     '',
      mealIngr:     '',
      judgeEnabled: $('custom-judge-toggle').checked,
      errorEl:      $('custom-form-error'),
      startBtn:     $('custom-start-btn'),
    });
  });

  /* Chat back */
  $('chat-back-btn').addEventListener('click', () => {
    if (S.sessionId && !S.ended) {
      $('confirm-modal').hidden = false;
    } else {
      goBackFromChat();
    }
  });

  /* Confirm modal */
  $('modal-cancel').addEventListener('click', () => { $('confirm-modal').hidden = true; });
  $('modal-confirm').addEventListener('click', async () => {
    $('confirm-modal').hidden = true;
    if (S.sessionId) {
      try { await fetch('/api/session/' + S.sessionId, { method: 'DELETE' }); } catch (_) {}
    }
    goBackFromChat();
  });

  /* Custom chat: send */
  $('send-btn').addEventListener('click', handleCustomSend);
  $('user-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!$('send-btn').disabled) handleCustomSend();
    }
  });
  $('user-input').addEventListener('input', () => {
    autoResize($('user-input'));
    $('send-btn').disabled = $('user-input').value.trim() === '' || S.ended;
  });
}

function goBackFromChat() {
  S.simRunning = false;
  S.sessionId  = null;
  S.ended      = false;
  $('messages').innerHTML = '';
  $('input-area').hidden  = true;
  $('sim-status-bar').hidden = true;
  show(S.prevConfig === 'sim' ? 'configSim' : 'configCustom');
}

/* ─── Config submit → start session ─────────────────────────── */
async function handleConfigSubmit({ mode, goal, mealType, mealDesc, mealIngr,
                                    judgeEnabled, errorEl, startBtn }) {
  hideError(errorEl);
  if (!goal) { showError(errorEl, 'Please select a nutritional goal.'); return; }
  if (mode === 'simulation' && !mealDesc) {
    showError(errorEl, 'Please describe the meal for the simulation.');
    return;
  }

  setLoading(startBtn, true);
  try {
    const res = await fetch('/api/session/start', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode,
        judge_enabled:    judgeEnabled,
        nutrition_goal:   goal,
        meal_type:        mealType,
        meal_description: mealDesc,
        meal_ingredient:  mealIngr,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Server error ' + res.status);
    }
    const data = await res.json();

    S.mode         = mode;
    S.judgeEnabled = judgeEnabled;
    S.sessionId    = data.session_id;
    S.ended        = false;
    S.prevConfig   = mode === 'simulation' ? 'sim' : 'custom';

    initChat({
      firstQuestion: data.first_question,
      goal:          fmtGoal(goal),
      mealDisplay:   mealDesc || fmtGoal(mealType),
      mode,
      judgeEnabled,
    });
  } catch (err) {
    showError(errorEl, 'Failed to start: ' + err.message);
  } finally {
    setLoading(startBtn, false);
  }
}

/* ─── Chat screen init ───────────────────────────────────────── */
function initChat({ firstQuestion, goal, mealDisplay, mode, judgeEnabled }) {
  const modePill = $('header-mode-pill');
  modePill.textContent = mode === 'simulation' ? 'Simulating Chat' : 'Custom Chat';
  modePill.className   = 'header-mode-pill ' + (mode === 'simulation' ? 'sim' : 'custom');

  $('header-goal').textContent = goal;
  $('header-meal').textContent = mealDisplay;
  const badge = $('judge-badge');
  badge.className  = judgeEnabled ? 'judge-badge judge-on' : 'judge-badge judge-off';
  $('judge-dot').style.display   = judgeEnabled ? '' : 'none';
  $('judge-badge-text').textContent = judgeEnabled ? 'Judge AI On' : 'Judge AI Off';

  $('messages').innerHTML = '';
  appendCoachMessage(firstQuestion);

  const inputArea = $('input-area');
  const simBar    = $('sim-status-bar');

  if (mode === 'simulation') {
    inputArea.hidden = true;
    simBar.hidden    = false;
    setSimStatus('Starting simulation\u2026');
  } else {
    inputArea.hidden = false;
    simBar.hidden    = true;
    const inp = $('user-input');
    inp.value    = '';
    inp.disabled = false;
    inp.style.height = '';
    $('send-btn').disabled = true;
    inp.focus();
  }

  show('chat');

  if (mode === 'simulation') runSimLoop();
}

/* ══════════════════════════════════════════════════════════════
   SIMULATION LOOP
══════════════════════════════════════════════════════════════ */
async function runSimLoop() {
  S.simRunning = true;
  setSimStatus('AI User is thinking\u2026');

  while (S.simRunning && !S.ended) {
    await sleep(900);
    if (!S.simRunning) break;

    const aiTyping = showTypingIndicator('ai-user');
    await sleep(700);

    let data;
    try {
      const res = await fetch('/api/session/' + S.sessionId + '/sim-step', { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Server error ' + res.status);
      }
      data = await res.json();
    } catch (err) {
      removeEl(aiTyping);
      appendSystemMessage('Error: ' + err.message);
      setSimStatus('Simulation stopped due to an error.');
      S.ended = true; S.simRunning = false;
      break;
    }

    removeEl(aiTyping);
    // user_reply가 비어있으면 버블 표시 생략 (TERMINATION_TOKEN만 생성된 경우)
    // Skip the AI User bubble if the reply is empty (user generated only the termination token)
    if (data.user_reply && data.user_reply.trim()) {
      appendAiUserMessage(data.user_reply, data.judge_aligned, data.aligned_label);
      await sleep(400);
    }

    // Coach 발화가 있으면 종료 여부와 무관하게 항상 표시
    // (stall_exit 시 Coach가 마무리 발화를 생성하므로 종료 전에 반드시 보여줘야 함)
    if (data.coach_question) {
      await sleep(600);
      const coachTyping = showTypingIndicator('coach');
      await sleep(600);
      removeEl(coachTyping);
      appendCoachMessage(data.coach_question);
      await sleep(400);
    }

    if (data.status !== 'active') {
      S.ended = true; S.simRunning = false;
      const msg = data.status === 'terminated' ? '\u2713 Simulation complete.'
                : data.status === 'max_turns'  ? 'Maximum turns reached. Simulation ended.'
                : 'Simulation ended.';
      appendSystemMessage(msg);
      setSimStatus('Simulation complete.');
      break;
    }

    setSimStatus('AI User is thinking\u2026');
  }
}

function setSimStatus(text) {
  $('sim-status-text').textContent = text;
}

/* ══════════════════════════════════════════════════════════════
   CUSTOM CHAT
══════════════════════════════════════════════════════════════ */
async function handleCustomSend() {
  const text = $('user-input').value.trim();
  if (!text || S.ended) return;

  const userWrap = appendUserMessage(text);
  $('user-input').value        = '';
  $('user-input').style.height = '';
  $('send-btn').disabled       = true;
  $('user-input').disabled     = true;

  const typing = showTypingIndicator('coach');

  try {
    const res = await fetch('/api/session/' + S.sessionId + '/turn', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ user_reply: text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Server error ' + res.status);
    }
    const data = await res.json();

    removeEl(typing);

    if (S.judgeEnabled) addAlignChip(userWrap, data.judge_aligned, data.aligned_label);

    if (data.status === 'active') {
      if (data.coach_question) appendCoachMessage(data.coach_question);
      $('user-input').disabled = false;
      $('send-btn').disabled   = true;
      $('user-input').focus();
    } else {
      S.ended = true;
      const msg = data.status === 'terminated' ? '\u2713 Session complete \u2014 goal alignment confirmed.'
                : data.status === 'max_turns'  ? 'Maximum turns reached. Session ended.'
                : 'Session ended.';
      appendSystemMessage(msg);
      $('user-input').disabled = true;
    }
  } catch (err) {
    removeEl(typing);
    appendSystemMessage('Error: ' + err.message);
    $('user-input').disabled = false;
    $('send-btn').disabled   = $('user-input').value.trim() === '';
  }
}

/* ══════════════════════════════════════════════════════════════
   MESSAGE BUILDERS
══════════════════════════════════════════════════════════════ */

function appendCoachMessage(text) {
  const row  = mkRow('coach');
  const av   = mkAvatar('Coach');
  const wrap = mkWrap();
  const lbl  = mkLabel(S.coachLabel || 'Coach');
  const bub  = mkBubble(text);
  wrap.appendChild(lbl);
  wrap.appendChild(bub);
  row.appendChild(av);
  row.appendChild(wrap);
  $('messages').appendChild(row);
  scrollBottom();
}

/** Returns the bubble-wrap so caller can attach chip later */
function appendUserMessage(text) {
  const row  = mkRow('user');
  const wrap = mkWrap();
  const lbl  = mkLabel('You');
  const bub  = mkBubble(text);
  wrap.appendChild(lbl);
  wrap.appendChild(bub);
  row.appendChild(wrap);
  $('messages').appendChild(row);
  scrollBottom();
  return wrap;
}

function appendAiUserMessage(text, judgeAligned, alignedLabel) {
  const row  = mkRow('ai-user');
  const av   = mkAvatar('User');
  const wrap = mkWrap();
  const lbl  = mkLabel(S.mode === 'simulation' ? (S.userLabel || 'AI User') : '');
  const bub  = mkBubble(text);
  wrap.appendChild(lbl);
  wrap.appendChild(bub);
  if (S.judgeEnabled && judgeAligned !== null && judgeAligned !== undefined) {
    addAlignChip(wrap, judgeAligned, alignedLabel);
  }
  row.appendChild(av);
  row.appendChild(wrap);
  $('messages').appendChild(row);
  scrollBottom();
}

function appendSystemMessage(text) {
  const row = mkRow('system');
  const bub = mkBubble(text);
  row.appendChild(bub);
  $('messages').appendChild(row);
  scrollBottom();
}

function addAlignChip(wrapEl, judgeAligned, alignedLabel) {
  const cls  = judgeAligned === null   ? 'pending'
             : judgeAligned            ? 'aligned'
             :                           'not-aligned';
  const chip = document.createElement('div');
  chip.className = 'align-chip ' + cls;

  const dot  = document.createElement('span');
  dot.className = 'align-chip-dot';
  chip.appendChild(dot);

  const txt  = document.createElement('span');
  txt.textContent = judgeAligned === null ? 'Evaluating\u2026'
                  : judgeAligned          ? '\u2713 Goal Aligned'
                  :                         '\u2717 Not Aligned';
  chip.appendChild(txt);
  wrapEl.appendChild(chip);
  scrollBottom();
  return chip;
}

function showTypingIndicator(role) {
  const row  = mkRow(role);

  if (role === 'coach')   row.appendChild(mkAvatar('Coach'));
  if (role === 'ai-user') row.appendChild(mkAvatar('User'));

  const wrap = mkWrap();
  const ind  = document.createElement('div');
  ind.className = 'typing-indicator';
  for (let i = 0; i < 3; i++) {
    const d = document.createElement('span');
    d.className = 'typing-dot';
    ind.appendChild(d);
  }
  wrap.appendChild(ind);
  row.appendChild(wrap);
  $('messages').appendChild(row);
  scrollBottom();
  return row;
}

/* ─── DOM micro-helpers ─────────────────────────────────────── */
function mkRow(cls) {
  const el = document.createElement('div');
  el.className = 'message ' + cls;
  return el;
}
function mkWrap() {
  const el = document.createElement('div');
  el.className = 'bubble-wrap';
  return el;
}
function mkBubble(text) {
  const el = document.createElement('div');
  el.className   = 'bubble';
  el.textContent = text;
  return el;
}
function mkLabel(text) {
  const el = document.createElement('div');
  el.className   = 'sender-label';
  el.textContent = text;
  return el;
}
function mkAvatar(text) {
  const el = document.createElement('div');
  el.className   = 'avatar';
  el.textContent = text;
  return el;
}

/* ─── Utility ───────────────────────────────────────────────── */
function scrollBottom() {
  const m = $('messages');
  m.scrollTop = m.scrollHeight;
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
function showError(el, msg) { el.textContent = msg; el.hidden = false; }
function hideError(el)      { el.hidden = true; }
function setLoading(btn, on) {
  btn.disabled = on;
  btn.querySelector('.btn-label').hidden   =  on;
  btn.querySelector('.btn-loading').hidden = !on;
}
function removeEl(el)  { if (el && el.parentNode) el.parentNode.removeChild(el); }
function sleep(ms)     { return new Promise(r => setTimeout(r, ms)); }
