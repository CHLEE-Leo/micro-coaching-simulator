/* ═══════════════════════════════════════════════════════════════
   Micro-Coaching Simulator  |  script.js  (v3)
   Landing → Meal Config → (User) → Coach → Alignment Tracker → Chat
   Wizard flow + turn-by-turn simulation
═══════════════════════════════════════════════════════════════ */

'use strict';

/* ─── App state ──────────────────────────────────────────────── */
const S = {
  screen:       'landing',
  mode:         null,         // 'simulation' | 'custom'
  sessionId:    null,
  ended:        false,
  turnIdx:      0,
  coachLabel:   null,
  userLabel:    null,
  chatgptAvailable: false,
  pendingCoachQ: null,        // buffered Coach question for next turn display

  /* Wizard data — accumulated across wizard steps */
  wizard: {
    goal:             '',
    mealType:         'dinner',
    mealDesc:         '',
    mealIngr:         '',
    userLlm:          'gemma',
    coachLlm:         'gemma',
    conversationMode: 'template-based',
    dialogSummarization: true,
    uncertaintyTracking: false,
    alignmentEnabled:     false,
    alignmentLlm:         'gemma',
    alignmentGoalDef:     true,
    alignmentWorkflow:    true,
    alignmentOutputFormat:'binary',
    personaPreferences:   [],
    personaAllergies:     [],
    personaRestrictions:  [],
  },
};

/* ─── DOM shortcuts ──────────────────────────────────────────── */
function $(id) { return document.getElementById(id); }

const screens = {
  landing:      null,
  configSim:    null,
  configCustom: null,
  configUser:   null,
  configCoach:  null,
  configAlignment:  null,
  chat:         null,
};

/* ─── Init ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  screens.landing      = $('landing-screen');
  screens.configSim    = $('config-sim-screen');
  screens.configCustom = $('config-custom-screen');
  screens.configUser   = $('config-user-screen');
  screens.configCoach  = $('config-coach-screen');
  screens.configAlignment  = $('config-alignment-screen');
  screens.chat         = $('chat-screen');

  loadGoals();
  bindEvents();
  pollServerReady();
});

/* ─── Server-ready polling ───────────────────────────────────── */
function pollServerReady() {
  const banner = document.createElement('div');
  banner.id = 'model-loading-banner';
  banner.textContent = '\u23F3  Loading AI model \u2014 first request will be ready shortly\u2026';
  document.body.appendChild(banner);

  const iv = setInterval(async () => {
    try {
      const data = await fetch('/api/status').then(r => r.json());
      if (data.ready) {
        if (data.coach_label) S.coachLabel = data.coach_label;
        if (data.user_label)  S.userLabel  = data.user_label;
        S.chatgptAvailable = !!data.chatgpt_available;
        ['user-llm', 'coach-llm', 'wizard-alignment-llm'].forEach(id => {
          const opt = document.querySelector('#' + id + ' option[value="chatgpt"]');
          if (opt) {
            opt.disabled = !S.chatgptAvailable;
            opt.textContent = S.chatgptAvailable ? 'GPT-5.2' : 'GPT-5.2 (unavailable)';
          }
        });
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


/* ═══════════════════════════════════════════════════════════════
   WIZARD STEPS INDICATOR
═══════════════════════════════════════════════════════════════ */
function getWizardSteps() {
  if (S.mode === 'simulation') return ['Meal', 'User', 'Coach', 'Alignment'];
  return ['Meal', 'Coach', 'Alignment'];
}

function renderWizardSteps(containerId, activeIdx) {
  const el = $(containerId);
  if (!el) return;
  const steps = getWizardSteps();
  el.innerHTML = '';
  steps.forEach((label, i) => {
    if (i > 0) {
      const line = document.createElement('div');
      line.className = 'wizard-step-line' + (i <= activeIdx ? ' done' : '');
      el.appendChild(line);
    }
    const step = document.createElement('div');
    step.className = 'wizard-step' + (i < activeIdx ? ' done' : '') + (i === activeIdx ? ' active' : '');
    const dot = document.createElement('div');
    dot.className = 'wizard-step-dot';
    dot.textContent = i < activeIdx ? '\u2713' : String(i + 1);
    const lbl = document.createElement('div');
    lbl.className = 'wizard-step-label';
    lbl.textContent = label;
    step.appendChild(dot);
    step.appendChild(lbl);
    el.appendChild(step);
  });
}


/* ═══════════════════════════════════════════════════════════════
   EVENT BINDINGS
═══════════════════════════════════════════════════════════════ */
function bindEvents() {
  /* ── Landing ── */
  $('pick-simulation').addEventListener('click', () => {
    S.mode = 'simulation';
    renderWizardSteps('sim-wizard-steps', 0);
    show('configSim');
  });
  $('pick-custom').addEventListener('click', () => {
    S.mode = 'custom';
    renderWizardSteps('custom-wizard-steps', 0);
    show('configCustom');
  });

  /* ── Back buttons ── */
  $('back-from-sim-config').addEventListener('click', () => show('landing'));
  $('back-from-custom-config').addEventListener('click', () => show('landing'));
  $('back-from-user-config').addEventListener('click', () => {
    renderWizardSteps('sim-wizard-steps', 0);
    show('configSim');
  });
  $('back-from-coach-config').addEventListener('click', () => {
    if (S.mode === 'simulation') {
      renderWizardSteps('user-wizard-steps', 1);
      $('config-user-split').classList.add('with-alignment-opts');
      refreshWizardUserPreview();
      show('configUser');
    } else {
      renderWizardSteps('custom-wizard-steps', 0);
      show('configCustom');
    }
  });
  $('back-from-alignment-config').addEventListener('click', () => {
    const idx = S.mode === 'simulation' ? 2 : 1;
    renderWizardSteps('coach-wizard-steps', idx);
    updateCoachModePill();
    $('config-coach-split').classList.add('with-alignment-opts');
    refreshWizardCoachPreview();
    show('configCoach');
  });

  /* ── Sim meal form → Next ── */
  $('sim-config-form').addEventListener('submit', e => {
    e.preventDefault();
    const goal = $('sim-goal').value;
    const desc = $('sim-meal-desc').value.trim();
    if (!goal) { showError($('sim-form-error'), 'Please select a nutritional goal.'); return; }
    if (!desc) { showError($('sim-form-error'), 'Please describe the meal.'); return; }
    hideError($('sim-form-error'));
    S.wizard.goal     = goal;
    S.wizard.mealType = $('sim-meal-type').value;
    S.wizard.mealDesc = desc;
    S.wizard.mealIngr = $('sim-meal-ingr').value.trim();
    renderWizardSteps('user-wizard-steps', 1);
    $('config-user-split').classList.add('with-alignment-opts');
    refreshWizardUserPreview();
    show('configUser');
  });

  /* ── Custom meal form → Next ── */
  $('custom-config-form').addEventListener('submit', e => {
    e.preventDefault();
    const goal = $('custom-goal').value;
    if (!goal) { showError($('custom-form-error'), 'Please select a nutritional goal.'); return; }
    hideError($('custom-form-error'));
    S.wizard.goal     = goal;
    S.wizard.mealType = $('custom-meal-type').value;
    S.wizard.mealDesc = '';
    S.wizard.mealIngr = '';
    renderWizardSteps('coach-wizard-steps', 1);
    updateCoachModePill();
    $('config-coach-split').classList.add('with-alignment-opts');
    refreshWizardCoachPreview();
    show('configCoach');
  });

  /* ── User config → Next ── */
  $('user-next-btn').addEventListener('click', () => {
    S.wizard.userLlm = $('user-llm').value;
    S.wizard.personaPreferences  = _csvToList($('persona-prefs'));
    S.wizard.personaAllergies    = _csvToList($('persona-allergy'));
    S.wizard.personaRestrictions = _csvToList($('persona-restrict'));
    renderWizardSteps('coach-wizard-steps', 2);
    updateCoachModePill();
    $('config-coach-split').classList.add('with-alignment-opts');
    refreshWizardCoachPreview();
    show('configCoach');
  });

  /* ── Coach config → Next ── */
  $('coach-next-btn').addEventListener('click', () => {
    S.wizard.coachLlm = $('coach-llm').value;
    S.wizard.dialogSummarization = $('dialog-summ-toggle').checked;
    S.wizard.uncertaintyTracking = $('uncertainty-toggle').checked;
    const alignmentIdx = S.mode === 'simulation' ? 3 : 2;
    renderWizardSteps('alignment-wizard-steps', alignmentIdx);
    updateAlignmentModePill();
    toggleAlignmentPreview($('wizard-alignment-toggle').checked);
    show('configAlignment');
  });

  /* ── Conversation mode cards ── */
  document.querySelectorAll('.conv-mode-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.conv-mode-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      S.wizard.conversationMode = card.dataset.mode;
      refreshWizardCoachPreview();
    });
  });

  /* ── Alignment Tracker wizard toggle ── */
  $('wizard-alignment-toggle').addEventListener('change', () => {
    const on = $('wizard-alignment-toggle').checked;
    $('wizard-alignment-body').hidden = !on;
    S.wizard.alignmentEnabled = on;
    toggleAlignmentPreview(on);
    if (on) refreshWizardAlignmentPreview();
  });
  $('wizard-alignment-goaldef').addEventListener('change', () => {
    S.wizard.alignmentGoalDef = $('wizard-alignment-goaldef').checked;
    refreshWizardAlignmentPreview();
  });
  $('wizard-alignment-workflow').addEventListener('change', () => {
    S.wizard.alignmentWorkflow = $('wizard-alignment-workflow').checked;
    refreshWizardAlignmentPreview();
  });
  $('wizard-alignment-format').addEventListener('change', () => {
    S.wizard.alignmentOutputFormat = $('wizard-alignment-format').value;
    refreshWizardAlignmentPreview();
  });
  $('wizard-alignment-llm').addEventListener('change', () => {
    S.wizard.alignmentLlm = $('wizard-alignment-llm').value;
  });

  /* ── Alignment Tracker wizard Start ── */
  $('wizard-start-btn').addEventListener('click', () => handleWizardStart());

  /* ── Chat back ── */
  $('chat-back-btn').addEventListener('click', () => {
    if (S.sessionId && !S.ended) {
      $('confirm-modal').hidden = false;
    } else {
      goBackFromChat();
    }
  });
  $('modal-cancel').addEventListener('click', () => { $('confirm-modal').hidden = true; });
  $('modal-confirm').addEventListener('click', async () => {
    $('confirm-modal').hidden = true;
    if (S.sessionId) {
      try { await fetch('/api/session/' + S.sessionId, { method: 'DELETE' }); } catch (_) {}
    }
    goBackFromChat();
  });

  /* ── Custom chat: send ── */
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

  /* ── Simulation: Next Turn button ── */
  $('sim-next-turn-btn').addEventListener('click', runSimSingleStep);

  /* ── Monitor panel tab switching ── */
  document.querySelectorAll('.monitor-tab').forEach(tab => {
    tab.addEventListener('click', () => switchMonitorTab(tab.dataset.tab));
  });
}


/* ─── Helper: mode pills for shared screens ─────────────────── */
function updateCoachModePill() {
  const p = $('coach-mode-pill');
  if (S.mode === 'simulation') {
    p.textContent = 'Simulating Chat';
    p.className = 'mode-pill sim';
  } else {
    p.textContent = 'Custom Chat';
    p.className = 'mode-pill custom';
  }
}
function updateAlignmentModePill() {
  const p = $('alignment-mode-pill');
  if (S.mode === 'simulation') {
    p.textContent = 'Simulating Chat';
    p.className = 'mode-pill sim';
  } else {
    p.textContent = 'Custom Chat';
    p.className = 'mode-pill custom';
  }
}

/* ─── Alignment Tracker preview panel toggle ─────────────────── */
function toggleAlignmentPreview(showIt) {
  const panel = $('wizard-alignment-preview-panel');
  const split = $('config-alignment-split');
  if (showIt) {
    panel.hidden = false;
    split.classList.add('with-alignment-opts');
  } else {
    panel.hidden = true;
    split.classList.remove('with-alignment-opts');
  }
}

function goBackFromChat() {
  S.sessionId     = null;
  S.ended         = false;
  S.turnIdx       = 0;
  S.pendingCoachQ = null;
  $('messages').innerHTML = '';
  $('input-area').hidden  = true;
  $('sim-status-bar').hidden = true;
  $('monitor-panel').hidden = true;
  $('alignment-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="alignment-panel-empty"><p>Alignment Tracker evaluation will appear here once enough conversation turns are collected.</p></div>';
  $('meal-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="meal-panel-empty"><p>Meal Fact Sheet will appear after MealTracker processes the conversation.</p></div>';
  $('certainty-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="certainty-panel-empty"><p>Certainty scores will appear when Uncertainty Tracking is enabled.</p></div>';
  $('orchestrator-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="orchestrator-panel-empty"><p>Orchestrator decisions and MealRecommender results will appear here.</p></div>';
  $('chat-body').classList.remove('with-alignment');
  const alignmentIdx = S.mode === 'simulation' ? 3 : 2;
  renderWizardSteps('alignment-wizard-steps', alignmentIdx);
  updateAlignmentModePill();
  toggleAlignmentPreview($('wizard-alignment-toggle').checked);
  show('configAlignment');
}

/* ─── Utility: CSV text → trimmed array ──────────────────────── */
function _csvToList(el) {
  if (!el || !el.value) return [];
  return el.value.split(',').map(s => s.trim()).filter(Boolean);
}


/* ═══════════════════════════════════════════════════════════════
   WIZARD START → Create session
═══════════════════════════════════════════════════════════════ */
function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function refreshWizardAlignmentPreview() {
  const goal = S.wizard.goal;
  const previewEl = $('wizard-alignment-prompt-preview');
  if (!previewEl) return;
  if (!goal) {
    previewEl.innerHTML = '<span style="color:var(--text-hint)">(Select a nutritional goal to see the prompt preview)</span>';
    return;
  }
  try {
    const params = new URLSearchParams({
      nutrition_goal: goal,
      goal_def:       S.wizard.alignmentGoalDef,
      workflow:       S.wizard.alignmentWorkflow,
      output_format:  S.wizard.alignmentOutputFormat,
    });
    const data = await fetch('/api/alignment-preview?' + params).then(r => r.json());
    const sysTxt = data.system_prompt || '';
    const usrTxt = data.user_message  || '';
    let html = escHtml(sysTxt);
    if (data.goal_def_text) {
      const esc = escHtml(data.goal_def_text);
      html = html.replace('- goal_definition: ' + esc,
        '<span class="preview-goaldef">- goal_definition: ' + esc + '</span>');
      const note = escHtml('(and goal_definition if available)');
      html = html.replace(note, '<span class="preview-goaldef">' + note + '</span>');
    }
    if (data.workflow_text) {
      const hdr = escHtml('WORKFLOW OF EXPERT NUTRITIONIST:');
      const bdy = escHtml(data.workflow_text);
      html = html.replace(hdr + '\n' + bdy,
        '<span class="preview-workflow">' + hdr + '\n' + bdy + '</span>');
    }
    if (data.output_format_text) {
      const esc = escHtml(data.output_format_text);
      html = html.replace(esc, '<span class="preview-outfmt">' + esc + '</span>');
    }
    html += '\n\n<span class="preview-separator">\u2500\u2500\u2500\u2500 User Message \u2500\u2500\u2500\u2500</span>\n\n';
    html += escHtml(usrTxt);
    previewEl.innerHTML = html;
  } catch (err) {
    previewEl.innerHTML = '<span style="color:#b03b3b">(Error loading preview: ' + escHtml(err.message) + ')</span>';
  }
}

async function refreshWizardCoachPreview() {
  const goal = S.wizard.goal;
  const previewEl = $('wizard-coach-prompt-preview');
  if (!previewEl) return;
  if (!goal) {
    previewEl.innerHTML = '<span style="color:var(--text-hint)">(Select a nutritional goal to see the prompt preview)</span>';
    return;
  }
  try {
    const params = new URLSearchParams({
      nutrition_goal:    goal,
      meal_type:         S.wizard.mealType || 'dinner',
      conversation_mode: S.wizard.conversationMode || 'template-based',
    });
    const data = await fetch('/api/coach-preview?' + params).then(r => r.json());
    let html = escHtml(data.system_prompt || '');
    if (data.action_guidelines_text) {
      const esc = escHtml(data.action_guidelines_text);
      html = html.replace(esc, '<span class="preview-workflow">' + esc + '</span>');
    }
    previewEl.innerHTML = html;
  } catch (err) {
    previewEl.innerHTML = '<span style="color:#b03b3b">(Error loading preview: ' + escHtml(err.message) + ')</span>';
  }
}

async function refreshWizardUserPreview() {
  const goal = S.wizard.goal;
  const previewEl = $('wizard-user-prompt-preview');
  if (!previewEl) return;
  if (!goal) {
    previewEl.innerHTML = '<span style="color:var(--text-hint)">(Select a nutritional goal to see the prompt preview)</span>';
    return;
  }
  try {
    const params = new URLSearchParams({
      nutrition_goal:    goal,
      meal_description:  S.wizard.mealDesc || '',
      meal_ingredient:   S.wizard.mealIngr || '',
    });
    const data = await fetch('/api/user-preview?' + params).then(r => r.json());
    previewEl.innerHTML = escHtml(data.system_prompt || '');
  } catch (err) {
    previewEl.innerHTML = '<span style="color:#b03b3b">(Error loading preview: ' + escHtml(err.message) + ')</span>';
  }
}

async function handleWizardStart() {
  const w = S.wizard;
  const errorEl  = $('wizard-form-error');
  const startBtn = $('wizard-start-btn');
  hideError(errorEl);
  setLoading(startBtn, true);

  try {
    const body = {
      mode:              S.mode,
      alignment_enabled:     w.alignmentEnabled,
      nutrition_goal:    w.goal,
      meal_type:         w.mealType,
      meal_description:  w.mealDesc,
      meal_ingredient:   w.mealIngr,
      llm_provider:      w.coachLlm || 'gemma',
      coach_conversation_mode: w.conversationMode,
      dialog_summarization: w.dialogSummarization,
      uncertainty_tracking: w.uncertaintyTracking,
    };
    if (w.personaPreferences.length)  body.persona_preferences  = w.personaPreferences;
    if (w.personaAllergies.length)    body.persona_allergies    = w.personaAllergies;
    if (w.personaRestrictions.length) body.persona_restrictions = w.personaRestrictions;
    if (S.mode === 'simulation') {
      body.user_llm_provider = w.userLlm || 'gemma';
    }
    if (w.alignmentEnabled) {
      body.alignment_use_goal_def  = w.alignmentGoalDef;
      body.alignment_use_workflow  = w.alignmentWorkflow;
      body.alignment_output_format = w.alignmentOutputFormat;
      body.alignment_llm_provider  = w.alignmentLlm || 'gemma';
    }

    const res = await fetch('/api/session/start', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Server error ' + res.status);
    }
    const data = await res.json();

    S.sessionId  = data.session_id;
    S.ended      = false;
    S.turnIdx    = 0;
    S.coachLabel = data.coach_label || 'Coach';
    S.userLabel  = data.user_label  || 'AI User';

    initChat({
      firstQuestion: data.first_question,
      goal:          fmtGoal(w.goal),
      mealDisplay:   w.mealDesc || fmtGoal(w.mealType),
      mode:          S.mode,
      alignmentEnabled:  w.alignmentEnabled,
    });
  } catch (err) {
    showError(errorEl, 'Failed to start: ' + err.message);
  } finally {
    setLoading(startBtn, false);
  }
}


/* ═══════════════════════════════════════════════════════════════
   CHAT SCREEN INIT
═══════════════════════════════════════════════════════════════ */
function initChat({ firstQuestion, goal, mealDisplay, mode, alignmentEnabled }) {
  const modePill = $('header-mode-pill');
  modePill.textContent = mode === 'simulation' ? 'Simulating Chat' : 'Custom Chat';
  modePill.className   = 'header-mode-pill ' + (mode === 'simulation' ? 'sim' : 'custom');

  $('header-goal').textContent = goal;
  $('header-meal').textContent = mealDisplay;

  // Determine if monitoring panel should be shown
  const monitorNeeded = true;  // MealTracker always runs; Alignment & UncertaintyTracker tabs auto-show
  const badge = $('monitor-badge');
  badge.className = monitorNeeded ? 'alignment-badge alignment-on' : 'alignment-badge alignment-off';
  $('monitor-dot').style.display    = monitorNeeded ? '' : 'none';
  $('monitor-badge-text').textContent = monitorNeeded ? 'Monitoring On' : 'Monitoring Off';

  S.turnIdx = 0;
  S.pendingCoachQ = firstQuestion;
  $('messages').innerHTML = '';

  if (mode === 'custom') {
    appendCoachMessage(firstQuestion, 0);
  }

  const monitorPanel = $('monitor-panel');
  const chatBody     = $('chat-body');
  if (monitorNeeded) {
    monitorPanel.hidden = false;
    chatBody.classList.add('with-alignment');
    // Show/hide tabs based on config
    setTabVisible('tab-alignment', alignmentEnabled);
    setTabVisible('tab-certainty', S.wizard.uncertaintyTracking);
    setTabVisible('tab-meal', true);  // MealTracker always runs
    setTabVisible('tab-orchestrator', true);  // Orchestrator always runs
    // Activate first visible tab
    activateFirstVisibleTab();
  } else {
    monitorPanel.hidden = true;
    chatBody.classList.remove('with-alignment');
  }

  // Reset monitoring content
  const alignmentCards = $('alignment-panel-cards');
  if (alignmentCards) alignmentCards.innerHTML = '<div class="alignment-panel-empty" id="alignment-panel-empty"><p>Alignment Tracker evaluation will appear here once enough conversation turns are collected.</p></div>';
  const mealCards = $('meal-panel-cards');
  if (mealCards) mealCards.innerHTML = '<div class="alignment-panel-empty" id="meal-panel-empty"><p>Meal Fact Sheet will appear after MealTracker processes the conversation.</p></div>';
  const certCards = $('certainty-panel-cards');
  if (certCards) certCards.innerHTML = '<div class="alignment-panel-empty" id="certainty-panel-empty"><p>Certainty scores will appear when Uncertainty Tracking is enabled.</p></div>';
  const orchCards = $('orchestrator-panel-cards');
  if (orchCards) orchCards.innerHTML = '<div class="alignment-panel-empty" id="orchestrator-panel-empty"><p>Orchestrator decisions and MealRecommender results will appear here.</p></div>';

  const inputArea = $('input-area');
  const simBar    = $('sim-status-bar');

  if (mode === 'simulation') {
    inputArea.hidden = true;
    simBar.hidden    = false;
    setSimStatus('Click "Next Turn" to proceed');
    $('sim-next-turn-btn').disabled = false;
    $('sim-status-dot').style.animation = 'none';
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
}


/* ═══════════════════════════════════════════════════════════════
   SIMULATION — TURN-BY-TURN
═══════════════════════════════════════════════════════════════ */
async function runSimSingleStep() {
  if (S.ended) return;
  const btn = $('sim-next-turn-btn');
  btn.disabled = true;
  $('sim-status-dot').style.animation = 'blink 1.4s infinite';

  // ── Step 1: Display buffered Coach question with typing indicator ──
  if (S.pendingCoachQ) {
    setSimStatus('Coach is thinking\u2026');
    const coachTyping = showTypingIndicator('coach');
    await sleep(600);
    removeEl(coachTyping);
    appendCoachMessage(S.pendingCoachQ, S.turnIdx);
    S.pendingCoachQ = null;
    await sleep(300);
  }

  // ── Step 2: Call sim-step for AI User response ──
  setSimStatus('AI User is thinking\u2026');
  const aiTyping = showTypingIndicator('ai-user');
  await sleep(500);

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
    S.ended = true;
    btn.disabled = true;
    return;
  }

  removeEl(aiTyping);

  if (data.user_reply && data.user_reply.trim()) {
    appendAiUserMessage(data.user_reply, data.alignment_aligned, data.aligned_label, data.turn_idx);
  }

  // Combined status row: alignment + certainty side-by-side
  appendStatusRow(data.turn_idx, data.alignment_aligned, data.alignment_score, data.aligned_label,
    data.certainty_score, data.certainty_reasoning);

  if (S.wizard.alignmentEnabled) {
    appendAlignmentCard(data.turn_idx, data.alignment_aligned, data.alignment_score,
      data.aligned_label, data.alignment_input, data.alignment_raw_output, data.alignment_reasoning);
  }

  // Update Backend Monitoring panel
  updateMonitorData(data);

  // ── Step 3: Buffer next Coach question for the next turn ──
  // Assessment double-turn: show assessment immediately, buffer preference Q
  if (data.assessment_message) {
    const assessTyping = showTypingIndicator('coach');
    await sleep(500);
    removeEl(assessTyping);
    appendCoachMessage(data.assessment_message, data.turn_idx + 1);
    await sleep(300);
  }
  if (data.coach_question) {
    S.pendingCoachQ = data.coach_question;
    S.turnIdx = data.assessment_message ? data.turn_idx + 2 : data.turn_idx + 1;
  }

  if (data.status !== 'active') {
    // Show final coach message immediately if there is one
    if (data.coach_question) {
      await sleep(400);
      const coachTyping = showTypingIndicator('coach');
      await sleep(500);
      removeEl(coachTyping);
      appendCoachMessage(data.coach_question, data.turn_idx + 1);
      S.pendingCoachQ = null;
    }
    S.ended = true;
    const msg = data.terminated_by === 'alignment'     ? '\u2713 Simulation complete — goal alignment confirmed.'
              : data.terminated_by === 'certainty'  ? '\u2713 Coach is confident about meal-goal alignment. Simulation ended.'
              : data.status === 'max_turns'          ? 'Maximum turns reached. Simulation ended.'
              : 'Simulation ended.';
    appendSystemMessage(msg);
    setSimStatus('Simulation complete.');
    $('sim-status-dot').style.animation = 'none';
    btn.disabled = true;
    showNextMealButton();
  } else {
    setSimStatus('Ready \u2014 click "Next Turn" to continue');
    $('sim-status-dot').style.animation = 'none';
    btn.disabled = false;
  }
}

function setSimStatus(text) {
  $('sim-status-text').textContent = text;
}


/* ═══════════════════════════════════════════════════════════════
   CUSTOM CHAT
═══════════════════════════════════════════════════════════════ */
async function handleCustomSend() {
  const text = $('user-input').value.trim();
  if (!text || S.ended) return;

  const userWrap = appendUserMessage(text, S.turnIdx);
  $('user-input').value        = '';
  $('user-input').style.height = '';
  $('send-btn').disabled       = true;
  $('user-input').disabled     = true;

  const typing = showTypingIndicator('coach');

  try {
    const res = await fetch('/api/session/' + S.sessionId + '/turn', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_reply: text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Server error ' + res.status);
    }
    const data = await res.json();

    removeEl(typing);

    // Guardrail blocked: 사용자 입력이 주제에서 벗어남 / User input off-topic
    if (data.guardrail_blocked) {
      if (data.coach_question) appendCoachMessage('\u26a0\ufe0f ' + data.coach_question, S.turnIdx);
      $('user-input').disabled = false;
      $('send-btn').disabled   = true;
      $('user-input').focus();
      return;
    }

    // Combined status row: alignment + certainty side-by-side
    appendStatusRow(data.turn_idx, data.alignment_aligned, data.alignment_score, data.aligned_label,
      data.certainty_score, data.certainty_reasoning);
    if (S.wizard.alignmentEnabled) {
      appendAlignmentCard(data.turn_idx, data.alignment_aligned, data.alignment_score,
        data.aligned_label, data.alignment_input, data.alignment_raw_output, data.alignment_reasoning);
    }

    // Update Backend Monitoring panel
    updateMonitorData(data);

    if (data.status === 'active') {
      S.turnIdx = data.assessment_message ? data.turn_idx + 2 : data.turn_idx + 1;
      // Assessment double-turn: show assessment first, then preference question
      if (data.assessment_message) {
        appendCoachMessage(data.assessment_message, data.turn_idx + 1);
        await sleep(400);
        const typing2 = showTypingIndicator('coach');
        await sleep(500);
        removeEl(typing2);
      }
      if (data.coach_question) appendCoachMessage(data.coach_question, S.turnIdx);
      $('user-input').disabled = false;
      $('send-btn').disabled   = true;
      $('user-input').focus();
    } else {
      S.ended = true;
      const msg = data.terminated_by === 'alignment'     ? '\u2713 Session complete \u2014 goal alignment confirmed.'
                : data.terminated_by === 'certainty'  ? '\u2713 Coach is confident about meal-goal alignment. Session ended.'
                : data.status === 'max_turns'          ? 'Maximum turns reached. Session ended.'
                : 'Session ended.';
      appendSystemMessage(msg);
      $('user-input').disabled = true;
      showNextMealButton();
    }
  } catch (err) {
    removeEl(typing);
    appendSystemMessage('Error: ' + err.message);
    $('user-input').disabled = false;
    $('send-btn').disabled   = $('user-input').value.trim() === '';
  }
}


/* ═══════════════════════════════════════════════════════════════
   MULTI-MEAL: Next Meal continuation
═══════════════════════════════════════════════════════════════ */
function showNextMealButton() {
  const wrap = document.createElement('div');
  wrap.className = 'next-meal-wrap';
  wrap.innerHTML =
    '<button class="next-meal-btn" onclick="handleNextMealPrompt()">' +
    '\ud83c\udf7d\ufe0f Start Next Meal (carry over preferences)' +
    '</button>';
  $('messages').appendChild(wrap);
  scrollBottom();
}

function handleNextMealPrompt() {
  const existing = document.querySelector('.next-meal-form');
  if (existing) return;

  const form = document.createElement('div');
  form.className = 'next-meal-form system-msg';
  form.innerHTML =
    '<div style="margin-bottom:6px;font-weight:600;">\ud83c\udf7d\ufe0f New Meal Setup</div>' +
    '<label>Meal type: <select id="next-meal-type">' +
    '<option value="breakfast">Breakfast</option>' +
    '<option value="lunch">Lunch</option>' +
    '<option value="dinner" selected>Dinner</option>' +
    '<option value="snack">Snack</option>' +
    '</select></label><br>' +
    '<label>Food items: <input id="next-meal-desc" type="text" placeholder="e.g. grilled chicken salad" style="width:220px"></label><br>' +
    '<label>Ingredients: <input id="next-meal-ingr" type="text" placeholder="(optional)" style="width:220px"></label><br>' +
    '<button id="next-meal-go-btn" style="margin-top:6px" onclick="handleNextMealStart()">Start</button>' +
    '<button style="margin-top:6px;margin-left:6px" onclick="this.closest(\'.next-meal-form\').remove()">Cancel</button>';
  $('messages').appendChild(form);
  scrollBottom();
}

async function handleNextMealStart() {
  const mealType = ($('next-meal-type') || {}).value || 'meal';
  const mealDesc = ($('next-meal-desc') || {}).value || '';
  const mealIngr = ($('next-meal-ingr') || {}).value || '';
  const goBtn = $('next-meal-go-btn');
  if (goBtn) goBtn.disabled = true;

  if (S.mode === 'simulation' && !mealDesc.trim()) {
    appendSystemMessage('Food items are required for simulation mode.');
    if (goBtn) goBtn.disabled = false;
    return;
  }

  try {
    const res = await fetch('/api/session/' + S.sessionId + '/continue', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nutrition_goal:   S.wizard.goal,
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

    S.sessionId  = data.session_id;
    S.ended      = false;
    S.turnIdx    = 0;
    S.pendingCoachQ = null;

    appendSystemMessage(
      '\ud83d\udd04 New meal session started (carrying over ' +
      data.previous_meals + ' previous meal' +
      (data.previous_meals !== 1 ? 's' : '') + ').'
    );

    $('header-meal').textContent = mealDesc || fmtGoal(mealType);

    if (S.mode === 'custom') {
      appendCoachMessage(data.first_question, 0);
      $('user-input').disabled = false;
      $('send-btn').disabled   = true;
      $('user-input').focus();
    } else {
      S.pendingCoachQ = data.first_question;
      const btn = $('sim-next-turn-btn');
      if (btn) {
        btn.disabled = false;
        setSimStatus('Ready \u2014 click "Next Turn" to continue');
        $('sim-status-dot').style.animation = 'none';
      }
    }
  } catch (err) {
    appendSystemMessage('Error starting next meal: ' + err.message);
    if (goBtn) goBtn.disabled = false;
  }
}


/* ═══════════════════════════════════════════════════════════════
   MESSAGE BUILDERS
═══════════════════════════════════════════════════════════════ */
function appendCoachMessage(text, turnIdx) {
  const row  = mkRow('coach');
  const av   = mkAvatar('Coach');
  const wrap = mkWrap();
  const hdr  = document.createElement('div');
  hdr.className = 'msg-header';
  hdr.appendChild(mkLabel(S.coachLabel || 'Coach'));
  if (turnIdx !== undefined && turnIdx !== null) hdr.appendChild(mkTurnBadge(turnIdx));
  const bub = mkBubble(text);
  wrap.appendChild(hdr);
  wrap.appendChild(bub);
  row.appendChild(av);
  row.appendChild(wrap);
  $('messages').appendChild(row);
  scrollBottom();
}

function appendUserMessage(text, turnIdx) {
  const row  = mkRow('user');
  const wrap = mkWrap();
  const hdr  = document.createElement('div');
  hdr.className = 'msg-header';
  hdr.appendChild(mkLabel('You'));
  if (turnIdx !== undefined && turnIdx !== null) hdr.appendChild(mkTurnBadge(turnIdx));
  const bub = mkBubble(text);
  wrap.appendChild(hdr);
  wrap.appendChild(bub);
  row.appendChild(wrap);
  $('messages').appendChild(row);
  scrollBottom();
  return wrap;
}

function appendAiUserMessage(text, alignmentAligned, alignedLabel, turnIdx) {
  const row  = mkRow('ai-user');
  const av   = mkAvatar('User');
  const wrap = mkWrap();
  const hdr  = document.createElement('div');
  hdr.className = 'msg-header';
  hdr.appendChild(mkLabel(S.mode === 'simulation' ? (S.userLabel || 'AI User') : ''));
  if (turnIdx !== undefined && turnIdx !== null) hdr.appendChild(mkTurnBadge(turnIdx));
  const bub = mkBubble(text);
  wrap.appendChild(hdr);
  wrap.appendChild(bub);
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

function appendStatusRow(turnIdx, alignmentAligned, alignmentScore, alignedLabel, certaintyScore, certaintyReasoning) {
  const hasAlign = S.wizard.alignmentEnabled && alignmentAligned !== null && alignmentAligned !== undefined;
  const hasCert  = certaintyScore !== null && certaintyScore !== undefined;
  if (!hasAlign && !hasCert) return;

  const row = document.createElement('div');
  row.className = 'message system status-row';

  if (hasAlign) {
    const scoreVal = (alignmentScore !== null && alignmentScore !== undefined) ? alignmentScore : (alignmentAligned ? 1.0 : 0.0);
    const high = scoreVal >= 0.5;
    const chip = document.createElement('div');
    chip.className = 'align-chip ' + (high ? 'aligned' : 'not-aligned');
    chip.innerHTML = '<span class="align-chip-icon">' + (high ? '🟢' : '🔴') + '</span>'
      + '<span>Alignment: ' + scoreVal.toFixed(2) + '</span>';
    row.appendChild(chip);
  }

  if (hasCert) {
    const high = certaintyScore >= 0.85;
    const chip = document.createElement('div');
    chip.className = 'certainty-chip ' + (high ? 'high' : 'low');
    chip.innerHTML = '<span class="certainty-icon">' + (high ? '💡' : '🤔') + '</span>'
      + '<span class="certainty-text">Certainty: ' + certaintyScore.toFixed(2) + '</span>';
    if (certaintyReasoning) chip.title = certaintyReasoning;
    row.appendChild(chip);
  }

  $('messages').appendChild(row);
  scrollBottom();
}

function showTypingIndicator(role) {
  const row = mkRow(role);
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
function mkTurnBadge(turnIdx) {
  const el = document.createElement('span');
  el.className   = 'turn-badge';
  el.textContent = 'Turn ' + turnIdx;
  return el;
}

/* ─── Utility ───────────────────────────────────────────────── */
function scrollBottom() {
  const m = $('messages');
  m.scrollTop = m.scrollHeight;
}
function scrollAlignmentBottom() {
  const p = $('alignment-panel-cards');
  if (p) p.scrollTop = p.scrollHeight;
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
function showError(el, msg) { el.textContent = msg; el.hidden = false; }
function hideError(el)      { el.hidden = true; }
function setLoading(btn, on) {
  btn.disabled = on;
  const lblEl  = btn.querySelector('.btn-label');
  if (lblEl) lblEl.hidden = on;
  const loadEl = btn.querySelector('.btn-loading');
  if (loadEl) loadEl.hidden = !on;
}
function removeEl(el) { if (el && el.parentNode) el.parentNode.removeChild(el); }
function sleep(ms)    { return new Promise(r => setTimeout(r, ms)); }


/* ─── Monitor panel helpers ─────────────────────────────────── */
function setTabVisible(tabId, visible) {
  const content = $(tabId);
  const tabBtn  = document.querySelector('.monitor-tab[data-tab="' + tabId + '"]');
  if (tabBtn) tabBtn.style.display = visible ? '' : 'none';
}

function activateFirstVisibleTab() {
  const tabs = document.querySelectorAll('.monitor-tab');
  for (const tabBtn of tabs) {
    if (tabBtn.style.display !== 'none') {
      switchMonitorTab(tabBtn.dataset.tab);
      return;
    }
  }
}

function switchMonitorTab(tabId) {
  document.querySelectorAll('.monitor-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.monitor-tab-content').forEach(c => c.classList.remove('active'));
  const btn = document.querySelector('.monitor-tab[data-tab="' + tabId + '"]');
  if (btn) btn.classList.add('active');
  const content = $(tabId);
  if (content) content.classList.add('active');
}

function updateMonitorData(data) {
  // Update Meal Tracker tab — per-turn card with input/output (same structure as Alignment Tracker)
  if (data.meal_tracker_output) {
    const empty = $('meal-panel-empty');
    if (empty) empty.hidden = true;

    const card = document.createElement('div');
    card.className = 'alignment-card';

    const header = document.createElement('div');
    header.className = 'alignment-card-header';
    const turnBadge = document.createElement('span');
    turnBadge.className = 'alignment-card-turn';
    turnBadge.textContent = 'Turn ' + data.turn_idx;
    header.appendChild(turnBadge);
    card.appendChild(header);

    if (data.meal_tracker_input) {
      const inputSection = document.createElement('div');
      inputSection.className = 'alignment-card-section';
      const inputToggle = document.createElement('button');
      inputToggle.className = 'alignment-card-toggle';
      inputToggle.textContent = '\u25B6 Input';
      inputToggle.type = 'button';
      const inputContent = document.createElement('pre');
      inputContent.className = 'alignment-card-pre collapsed';
      inputContent.textContent = data.meal_tracker_input;
      inputToggle.addEventListener('click', () => {
        const collapsed = inputContent.classList.toggle('collapsed');
        inputToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Input';
      });
      inputSection.appendChild(inputToggle);
      inputSection.appendChild(inputContent);
      card.appendChild(inputSection);
    }

    const outputSection = document.createElement('div');
    outputSection.className = 'alignment-card-section';
    const outputToggle = document.createElement('button');
    outputToggle.className = 'alignment-card-toggle';
    outputToggle.textContent = '\u25BC Output (Meal Fact Sheet)';
    outputToggle.type = 'button';
    const outputContent = document.createElement('pre');
    outputContent.className = 'alignment-card-pre';
    outputContent.textContent = data.meal_tracker_output;
    outputToggle.addEventListener('click', () => {
      const collapsed = outputContent.classList.toggle('collapsed');
      outputToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Output (Meal Fact Sheet)';
    });
    outputSection.appendChild(outputToggle);
    outputSection.appendChild(outputContent);
    card.appendChild(outputSection);

    const container = $('meal-panel-cards');
    if (container) { container.appendChild(card); container.scrollTop = container.scrollHeight; }
  }

  // Update Uncertainty Tracker tab — per-turn card with input/output
  if (data.certainty_score !== null && data.certainty_score !== undefined) {
    const empty = $('certainty-panel-empty');
    if (empty) empty.hidden = true;

    const card = document.createElement('div');
    card.className = 'alignment-card';

    const header = document.createElement('div');
    header.className = 'alignment-card-header';
    const turnBadge = document.createElement('span');
    turnBadge.className = 'alignment-card-turn';
    turnBadge.textContent = 'Turn ' + data.turn_idx;
    header.appendChild(turnBadge);
    const high = data.certainty_score >= 0.85;
    const verdict = document.createElement('span');
    verdict.className = 'alignment-card-verdict ' + (high ? 'aligned' : 'not-aligned');
    verdict.textContent = (high ? '\uD83C\uDFAF ' : '\uD83D\uDD0D ') + 'Certainty: ' + data.certainty_score.toFixed(2);
    header.appendChild(verdict);
    card.appendChild(header);

    if (data.certainty_reasoning) {
      const reasonLine = document.createElement('div');
      reasonLine.className = 'alignment-card-score';
      reasonLine.textContent = data.certainty_reasoning;
      card.appendChild(reasonLine);
    }

    if (data.certainty_input) {
      const inputSection = document.createElement('div');
      inputSection.className = 'alignment-card-section';
      const inputToggle = document.createElement('button');
      inputToggle.className = 'alignment-card-toggle';
      inputToggle.textContent = '\u25B6 Input';
      inputToggle.type = 'button';
      const inputContent = document.createElement('pre');
      inputContent.className = 'alignment-card-pre collapsed';
      inputContent.textContent = data.certainty_input;
      inputToggle.addEventListener('click', () => {
        const collapsed = inputContent.classList.toggle('collapsed');
        inputToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Input';
      });
      inputSection.appendChild(inputToggle);
      inputSection.appendChild(inputContent);
      card.appendChild(inputSection);
    }

    if (data.certainty_output) {
      const outputSection = document.createElement('div');
      outputSection.className = 'alignment-card-section';
      const outputToggle = document.createElement('button');
      outputToggle.className = 'alignment-card-toggle';
      outputToggle.textContent = '\u25B6 Raw Output';
      outputToggle.type = 'button';
      const outputContent = document.createElement('pre');
      outputContent.className = 'alignment-card-pre collapsed';
      outputContent.textContent = data.certainty_output;
      outputToggle.addEventListener('click', () => {
        const collapsed = outputContent.classList.toggle('collapsed');
        outputToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Raw Output';
      });
      outputSection.appendChild(outputToggle);
      outputSection.appendChild(outputContent);
      card.appendChild(outputSection);
    }

    const container = $('certainty-panel-cards');
    if (container) { container.appendChild(card); container.scrollTop = container.scrollHeight; }
  }

  // Update Orchestrator tab — decision + recommendation + assessment per turn
  if (data.orchestrator_decision) {
    const empty = $('orchestrator-panel-empty');
    if (empty) empty.hidden = true;

    const card = document.createElement('div');
    card.className = 'alignment-card';

    const header = document.createElement('div');
    header.className = 'alignment-card-header';
    const turnBadge = document.createElement('span');
    turnBadge.className = 'alignment-card-turn';
    turnBadge.textContent = 'Turn ' + data.turn_idx;
    header.appendChild(turnBadge);

    const dec = data.orchestrator_decision;
    const phaseLabels = {
      'info_seeking':        'Information Seeking',
      'assessment':          'Assessment',
      'rec_info_seeking':    'Rec Info Seeking',
      'recommending':        'Recommending',
      'negotiation':         'Negotiation',
      'motivational_ending': 'Motivational Ending',
      'terminated':          'Terminated',
    };
    const phaseCls = (data.phase === 'terminated' || data.phase === 'motivational_ending') ? 'aligned'
                   : 'pending';
    const phaseBadge = document.createElement('span');
    phaseBadge.className = 'alignment-card-verdict ' + phaseCls;
    phaseBadge.textContent = phaseLabels[data.phase] || data.phase;
    header.appendChild(phaseBadge);
    card.appendChild(header);

    const actionLabels = {
      'terminate':              '\u26D4 Terminate',
      'recommend':              '\uD83D\uDCA1 Recommend',
      'assess_meal':            '\uD83D\uDCCB Assess Meal',
      'seek_recommendation_info': '\uD83D\uDD0E Seek Rec Info',
      'seek_meal_info':         '\uD83D\uDD0D Seek Meal Info',
      'motivational_close':     '\u2728 Motivational Close',
    };
    const actionLine = document.createElement('div');
    actionLine.className = 'alignment-card-score';
    actionLine.style.fontWeight = '600';
    actionLine.textContent = 'Action: ' + (actionLabels[dec.action] || dec.action);
    card.appendChild(actionLine);

    if (dec.reasoning) {
      const reasonLine = document.createElement('div');
      reasonLine.className = 'alignment-card-score';
      reasonLine.textContent = dec.reasoning;
      card.appendChild(reasonLine);
    }

    if (dec.instruction && (dec.action === 'recommend' || dec.action === 'motivational_close' || dec.action === 'terminate')) {
      const instrLine = document.createElement('div');
      instrLine.className = 'alignment-card-score';
      instrLine.style.fontStyle = 'italic';
      instrLine.style.color = '#666';
      instrLine.textContent = '\u2192 Instruction: ' + dec.instruction;
      card.appendChild(instrLine);
    }

    // Show assessment result if present
    if (data.assessment_result) {
      const assess = data.assessment_result;
      const assessSection = document.createElement('div');
      assessSection.className = 'alignment-card-section';
      const assessToggle = document.createElement('button');
      assessToggle.className = 'alignment-card-toggle';
      assessToggle.textContent = '\u25BC Assessment';
      assessToggle.type = 'button';
      const assessContent = document.createElement('pre');
      assessContent.className = 'alignment-card-pre';
      assessContent.textContent = 'Overall: ' + (assess.overall || 'N/A')
        + '\nStrengths: ' + (assess.strengths || 'N/A')
        + '\nGaps: ' + (assess.gaps || 'N/A')
        + '\nSuggestion: ' + (assess.suggestion || 'N/A');
      assessToggle.addEventListener('click', () => {
        const collapsed = assessContent.classList.toggle('collapsed');
        assessToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Assessment';
      });
      assessSection.appendChild(assessToggle);
      assessSection.appendChild(assessContent);
      card.appendChild(assessSection);
    }

    // Show recommendation result if present
    if (data.recommendation_result) {
      const rec = data.recommendation_result;
      const recSection = document.createElement('div');
      recSection.className = 'alignment-card-section';
      const recToggle = document.createElement('button');
      recToggle.className = 'alignment-card-toggle';
      recToggle.textContent = '\u25BC Recommendation';
      recToggle.type = 'button';
      const recContent = document.createElement('pre');
      recContent.className = 'alignment-card-pre';
      recContent.textContent = 'Type: ' + (rec.recommendation_type || 'N/A')
        + '\nTarget: ' + (rec.target_food || 'N/A')
        + '\nSuggestion: ' + (rec.suggestion || 'N/A')
        + '\nReasoning: ' + (rec.reasoning || 'N/A')
        + '\nExpected Impact: ' + (rec.expected_impact || 'N/A');
      recToggle.addEventListener('click', () => {
        const collapsed = recContent.classList.toggle('collapsed');
        recToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Recommendation';
      });
      recSection.appendChild(recToggle);
      recSection.appendChild(recContent);
      card.appendChild(recSection);
    }

    // Show Router input (collapsed by default)
    if (data.orchestrator_input) {
      const inputSection = document.createElement('div');
      inputSection.className = 'alignment-card-section';
      const inputToggle = document.createElement('button');
      inputToggle.className = 'alignment-card-toggle';
      inputToggle.textContent = '\u25B6 Input (Router Prompt)';
      inputToggle.type = 'button';
      const inputContent = document.createElement('pre');
      inputContent.className = 'alignment-card-pre collapsed';
      inputContent.textContent = data.orchestrator_input;
      inputToggle.addEventListener('click', () => {
        const collapsed = inputContent.classList.toggle('collapsed');
        inputToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Input (Router Prompt)';
      });
      inputSection.appendChild(inputToggle);
      inputSection.appendChild(inputContent);
      card.appendChild(inputSection);
    }

    // Show Router raw output (collapsed by default)
    if (data.orchestrator_raw_output !== null && data.orchestrator_raw_output !== undefined) {
      const outputSection = document.createElement('div');
      outputSection.className = 'alignment-card-section';
      const outputToggle = document.createElement('button');
      outputToggle.className = 'alignment-card-toggle';
      outputToggle.textContent = '\u25B6 Raw Output';
      outputToggle.type = 'button';
      const outputContent = document.createElement('pre');
      outputContent.className = 'alignment-card-pre collapsed';
      outputContent.textContent = data.orchestrator_raw_output || '(empty)';
      outputToggle.addEventListener('click', () => {
        const collapsed = outputContent.classList.toggle('collapsed');
        outputToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Raw Output';
      });
      outputSection.appendChild(outputToggle);
      outputSection.appendChild(outputContent);
      card.appendChild(outputSection);
    }

    const orchContainer = $('orchestrator-panel-cards');
    if (orchContainer) { orchContainer.appendChild(card); orchContainer.scrollTop = orchContainer.scrollHeight; }
  }
}


/* ═══════════════════════════════════════════════════════════════
   ALIGNMENT TRACKER PANEL CARD BUILDER
═══════════════════════════════════════════════════════════════ */
function appendAlignmentCard(turnIdx, alignmentAligned, alignmentScore, alignedLabel, alignmentInput, alignmentRawOutput, alignmentReasoning) {
  const empty = $('alignment-panel-empty');
  if (empty) empty.hidden = true;

  const card = document.createElement('div');
  card.className = 'alignment-card';

  /* Header */
  const header = document.createElement('div');
  header.className = 'alignment-card-header';

  const turnBadge = document.createElement('span');
  turnBadge.className = 'alignment-card-turn';
  turnBadge.textContent = 'Turn ' + turnIdx;
  header.appendChild(turnBadge);

  const chipCls = alignmentAligned === null ? 'pending'
                : alignmentAligned          ? 'aligned'
                :                         'not-aligned';
  const verdict = document.createElement('span');
  verdict.className = 'alignment-card-verdict ' + chipCls;
  verdict.textContent = alignmentAligned === null ? 'Pending'
                      : alignmentAligned          ? 'Aligned'
                      :                         'Not Aligned';
  header.appendChild(verdict);
  card.appendChild(header);

  /* Score */
  if (alignmentScore !== null && alignmentScore !== undefined) {
    const scoreLine = document.createElement('div');
    scoreLine.className = 'alignment-card-score';
    scoreLine.textContent = 'Score: ' + alignmentScore.toFixed(2);
    card.appendChild(scoreLine);
  }

  /* Reasoning */
  if (alignmentReasoning) {
    const reasonLine = document.createElement('div');
    reasonLine.className = 'alignment-card-score';
    reasonLine.textContent = alignmentReasoning;
    card.appendChild(reasonLine);
  }

  /* Input transcript (collapsible) */
  if (alignmentInput) {
    const inputSection = document.createElement('div');
    inputSection.className = 'alignment-card-section';
    const inputToggle = document.createElement('button');
    inputToggle.className = 'alignment-card-toggle';
    inputToggle.textContent = '\u25B6 Input (transcript)';
    inputToggle.type = 'button';
    const inputContent = document.createElement('pre');
    inputContent.className = 'alignment-card-pre collapsed';
    inputContent.textContent = alignmentInput;
    inputToggle.addEventListener('click', () => {
      const collapsed = inputContent.classList.toggle('collapsed');
      inputToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Input (transcript)';
    });
    inputSection.appendChild(inputToggle);
    inputSection.appendChild(inputContent);
    card.appendChild(inputSection);
  }

  /* Raw output (collapsible) */
  if (alignmentRawOutput) {
    const outputSection = document.createElement('div');
    outputSection.className = 'alignment-card-section';
    const outputToggle = document.createElement('button');
    outputToggle.className = 'alignment-card-toggle';
    outputToggle.textContent = '\u25B6 Raw Output';
    outputToggle.type = 'button';
    const outputContent = document.createElement('pre');
    outputContent.className = 'alignment-card-pre collapsed';
    outputContent.textContent = alignmentRawOutput;
    outputToggle.addEventListener('click', () => {
      const collapsed = outputContent.classList.toggle('collapsed');
      outputToggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' Raw Output';
    });
    outputSection.appendChild(outputToggle);
    outputSection.appendChild(outputContent);
    card.appendChild(outputSection);
  }

  /* No data yet */
  if (!alignmentInput && !alignmentRawOutput) {
    const noData = document.createElement('div');
    noData.className = 'alignment-card-no-data';
    noData.textContent = 'Not enough turns for evaluation yet.';
    card.appendChild(noData);
  }

  $('alignment-panel-cards').appendChild(card);
  scrollAlignmentBottom();
}
