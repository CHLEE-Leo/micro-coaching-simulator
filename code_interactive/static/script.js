/* ═══════════════════════════════════════════════════════════════
   Micro-Coaching  |  script.js
   Landing -> Meal Config -> Coach -> Chat
═══════════════════════════════════════════════════════════════ */

'use strict';

/* ─── App state ──────────────────────────────────────────────── */
const S = {
  screen:       'landing',
  mode:         null,         // 'custom' | 'deploy'
  sessionId:    null,
  ended:        false,
  turnIdx:      0,
  coachLabel:   null,
  userLabel:    null,
  pendingCoachQ: null,        // buffered Coach question for next turn display
  _prevFocus:   null,         // focus element before modal open

  /* Wizard data — accumulated across wizard steps */
  wizard: {
    goal:             '',
    mealType:         'dinner',
    mealDesc:         '',
    mealIngr:         '',
    conversationMode: 'template-based',
    contextTracking: true,
    uncertaintyTracking: false,
    alignmentEnabled:     false,
    alignmentGoalDef:     true,
    alignmentWorkflow:    true,
    alignmentOutputFormat:'0-1',
    personaActivityLevel: '',
    personaDietPrefs:     [],
    personaAllergies:     [],
    personaHealthConcerns:[],
  },
};

/* ─── DOM shortcuts ──────────────────────────────────────────── */
function $(id) { return document.getElementById(id); }

const screens = {
  landing:      null,
  configCustom: null,
  configDeploy: null,
  configCoach:  null,
  configAlignment:  null,
  chat:         null,
};

/* ─── Init ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  screens.landing      = $('landing-screen');
  screens.configCustom = $('config-custom-screen');
  screens.configDeploy = $('config-deploy-screen');
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
        banner.textContent = '\u2713  AI model ready!';
        banner.style.background = 'linear-gradient(135deg, #10b981, #059669)';
        setTimeout(() => banner.remove(), 2500);
        clearInterval(iv);
      }
    } catch (_) {}
  }, 3000);
}

/* ─── Load goals into config selects ──────────────────────────── */
async function loadGoals() {
  let goals = [];
  try {
    const data = await fetch('/api/goals').then(r => r.json());
    goals = data.goals || [];
  } catch (_) {}

  ['custom-goal', 'deploy-goal'].forEach(selId => {
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
  return ['Meal', 'Coach', 'Estimator'];
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
    dot.setAttribute('aria-label', 'Step ' + (i + 1) + ': ' + label + (i < activeIdx ? ' (complete)' : i === activeIdx ? ' (current)' : ''));
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
  $('pick-custom').addEventListener('click', () => {
    S.mode = 'custom';
    // Custom mode: single-step wizard, no step indicator needed
    const stepsEl = $('custom-wizard-steps');
    if (stepsEl) stepsEl.innerHTML = '';
    show('configCustom');
  });
  $('pick-deploy').addEventListener('click', () => {
    S.mode = 'deploy';
    show('configDeploy');
  });

  /* ── Back buttons ── */
  $('back-from-custom-config').addEventListener('click', () => show('landing'));
  $('back-from-deploy-config').addEventListener('click', () => show('landing'));
  $('back-from-coach-config').addEventListener('click', () => {
    renderWizardSteps('custom-wizard-steps', 0);
    show(S.mode === 'deploy' ? 'configDeploy' : 'configCustom');
  });
  $('back-from-alignment-config').addEventListener('click', () => {
    renderWizardSteps('coach-wizard-steps', 1);
    updateCoachModePill();
    $('config-coach-split').classList.add('with-alignment-opts');
    refreshWizardCoachPreview();
    show('configCoach');
  });

  /* ── Custom meal form → Start Session directly ── */
  $('custom-config-form').addEventListener('submit', async e => {
    e.preventDefault();
    const goal = $('custom-goal').value;
    if (!goal) { showError($('custom-form-error'), 'Please select a nutritional goal.'); return; }
    hideError($('custom-form-error'));
    S.wizard.goal     = goal;
    S.wizard.mealType = $('custom-meal-type').value;
    S.wizard.mealDesc = '';
    S.wizard.mealIngr = '';
    // Collect profile from collapsible section
    S.wizard.personaActivityLevel  = $('custom-activity').value;
    S.wizard.personaDietPrefs      = _collectChips('custom-diet');
    S.wizard.personaAllergies      = _collectChips('custom-allergy');
    S.wizard.personaHealthConcerns = _collectChips('custom-health');
    // Custom mode defaults
    S.wizard.conversationMode   = 'open-ended';
    S.wizard.contextTracking = true;
    S.wizard.uncertaintyTracking = false;
    S.wizard.alignmentEnabled    = false;
    S.wizard.alignmentGoalDef    = true;
    S.wizard.alignmentWorkflow   = true;
    S.wizard.alignmentOutputFormat = '0-1';
    await handleWizardStart({ errorEl: $('custom-form-error'), startBtn: $('custom-next-btn') });
  });

  /* ── Deploy meal form → Start Session directly (no monitoring) ── */
  $('deploy-config-form').addEventListener('submit', async e => {
    e.preventDefault();
    const goal = $('deploy-goal').value;
    if (!goal) { showError($('deploy-form-error'), 'Please select a nutritional goal.'); return; }
    hideError($('deploy-form-error'));
    S.wizard.goal     = goal;
    S.wizard.mealType = $('deploy-meal-type').value;
    S.wizard.mealDesc = '';
    S.wizard.mealIngr = '';
    // Collect profile from collapsible section
    S.wizard.personaActivityLevel  = $('deploy-activity').value;
    S.wizard.personaDietPrefs      = _collectChips('deploy-diet');
    S.wizard.personaAllergies      = _collectChips('deploy-allergy');
    S.wizard.personaHealthConcerns = _collectChips('deploy-health');
    // Deploy mode defaults (same pipeline as custom, but UI hides all monitoring)
    S.wizard.conversationMode   = 'open-ended';
    S.wizard.contextTracking = true;
    S.wizard.uncertaintyTracking = false;
    S.wizard.alignmentEnabled    = false;
    S.wizard.alignmentGoalDef    = true;
    S.wizard.alignmentWorkflow   = true;
    S.wizard.alignmentOutputFormat = '0-1';
    await handleWizardStart({ errorEl: $('deploy-form-error'), startBtn: $('deploy-next-btn') });
  });

  /* ── Coach config → Next ── */
  $('coach-next-btn').addEventListener('click', () => {
    S.wizard.contextTracking = $('dialog-summ-toggle').checked;
    renderWizardSteps('alignment-wizard-steps', 2);
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

  /* ── Alignment Tracker wizard Start ── */
  $('wizard-start-btn').addEventListener('click', () => {
    S.wizard.uncertaintyTracking = $('uncertainty-toggle').checked;
    handleWizardStart();
  });

  /* ── Chat back ── */
  $('chat-back-btn').addEventListener('click', () => {
    if (S.sessionId && !S.ended) {
      S._prevFocus = document.activeElement;
      $('confirm-modal').hidden = false;
      $('modal-cancel').focus();
    } else {
      goBackFromChat();
    }
  });
  $('modal-cancel').addEventListener('click', () => { closeModal(); });
  $('modal-confirm').addEventListener('click', async () => {
    closeModal();
    if (S.sessionId) {
      try { await fetch('/api/session/' + S.sessionId, { method: 'DELETE' }); } catch (_) {}
    }
    goBackFromChat();
  });

  /* ── Modal: close on Escape key ── */
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('confirm-modal').hidden) {
      closeModal();
    }
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

  /* ── Monitor panel tab switching ── */
  document.querySelectorAll('.monitor-tab').forEach(tab => {
    tab.addEventListener('click', () => switchMonitorTab(tab.dataset.tab));
  });

  /* ── Export chat history button ── */
  $('export-chat-btn').addEventListener('click', exportChatHistory);
}


/* ─── Export Chat History ────────────────────────────────────── */
async function exportChatHistory() {
  if (!S.sessionId) return;
  const btn = $('export-chat-btn');
  btn.disabled = true;
  try {
    const res = await fetch('/api/session/' + encodeURIComponent(S.sessionId) + '/history');
    if (!res.ok) throw new Error('Failed to fetch history');
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chat_history_' + S.sessionId.slice(0, 8) + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error('Export failed:', e);
  } finally {
    btn.disabled = false;
  }
}


/* ─── Helper: mode pills for shared screens ─────────────────── */
function updateCoachModePill() {
  const p = $('coach-mode-pill');
  if (S.mode === 'deploy') {
    p.textContent = 'Deploy Chat';
    p.className = 'mode-pill deploy';
  } else {
    p.textContent = 'Custom Chat';
    p.className = 'mode-pill custom';
  }
}
function updateAlignmentModePill() {
  const p = $('alignment-mode-pill');
  if (S.mode === 'deploy') {
    p.textContent = 'Deploy Chat';
    p.className = 'mode-pill deploy';
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
  $('monitor-panel').hidden = true;
  $('export-chat-btn').hidden = true;
  $('alignment-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="alignment-panel-empty"><p>Alignment Tracker evaluation will appear here once enough conversation turns are collected.</p></div>';
  $('meal-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="meal-panel-empty"><p>Meal Base will appear after MealTracker processes the conversation.</p></div>';
  $('certainty-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="certainty-panel-empty"><p>Certainty scores will appear when Uncertainty Tracking is enabled.</p></div>';
  $('planner-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="planner-panel-empty"><p>Dialogue Planner decisions and Meal Recommender results will appear here.</p></div>';
  $('guardrail-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="guardrail-panel-empty"><p>Guardrail input/output will appear here for each turn.</p></div>';
  $('context-panel-cards').innerHTML = '<div class="alignment-panel-empty" id="context-panel-empty"><p>Context Base will appear after ContextTracker processes the conversation.</p></div>';
  $('chat-body').classList.remove('with-alignment');
  if (S.mode === 'custom') {
    renderWizardSteps('custom-wizard-steps', 0);
    show('configCustom');
  } else if (S.mode === 'deploy') {
    show('configDeploy');
  } else {
    renderWizardSteps('alignment-wizard-steps', 2);
    updateAlignmentModePill();
    toggleAlignmentPreview($('wizard-alignment-toggle').checked);
    show('configAlignment');
  }
}

/* ─── Utility: CSV text → trimmed array ──────────────────────── */
function _csvToList(el) {
  if (!el || !el.value) return [];
  return el.value.split(',').map(s => s.trim()).filter(Boolean);
}

/* ─── Utility: collect selected chips from a chip-select container ── */
function _collectChips(containerId) {
  const el = $(containerId);
  if (!el) return [];
  return Array.from(el.querySelectorAll('.chip.selected'))
              .map(c => c.dataset.value);
}

/* ─── Chip toggle: delegate click on all .chip-select containers ── */
document.addEventListener('click', e => {
  const chip = e.target.closest('.chip-select .chip');
  if (chip) chip.classList.toggle('selected');
});


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
      output_format:  '0-1',
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

async function handleWizardStart(opts) {
  opts = opts || {};
  const w = S.wizard;
  const errorEl  = opts.errorEl  || $('wizard-form-error');
  const startBtn = opts.startBtn || $('wizard-start-btn');
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
      coach_conversation_mode: w.conversationMode,
      context_tracking: w.contextTracking,
      uncertainty_tracking: w.uncertaintyTracking,
    };
    if (w.personaActivityLevel)           body.persona_activity_level  = w.personaActivityLevel;
    if (w.personaDietPrefs.length)        body.persona_diet_preferences = w.personaDietPrefs;
    if (w.personaAllergies.length)        body.persona_allergies        = w.personaAllergies;
    if (w.personaHealthConcerns.length)   body.persona_health_concerns  = w.personaHealthConcerns;
    if (w.alignmentEnabled) {
      body.alignment_use_goal_def  = w.alignmentGoalDef;
      body.alignment_use_workflow  = w.alignmentWorkflow;
      body.alignment_output_format = w.alignmentOutputFormat;

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
  if (mode === 'deploy') {
    modePill.textContent = 'Deploy Chat';
    modePill.className   = 'header-mode-pill deploy';
  } else {
    modePill.textContent = 'Custom Chat';
    modePill.className   = 'header-mode-pill custom';
  }

  $('header-goal').textContent = goal;
  $('header-meal').textContent = mealDisplay;

  // Show export button for non-deploy modes
  $('export-chat-btn').hidden = (mode === 'deploy');

  // Determine if monitoring panel should be shown.
  const monitorNeeded = mode === 'custom';
  const badge = $('monitor-badge');
  if (mode === 'deploy') {
    // Deploy: completely hide monitoring badge
    badge.hidden = true;
  } else {
    badge.className = monitorNeeded ? 'alignment-badge alignment-on' : 'alignment-badge alignment-off';
    $('monitor-dot').style.display    = monitorNeeded ? '' : 'none';
    $('monitor-badge-text').textContent = monitorNeeded ? 'Monitoring On' : 'Monitoring Off';
    badge.hidden = !monitorNeeded;
  }

  S.turnIdx = 0;
  S.pendingCoachQ = firstQuestion;
  $('messages').innerHTML = '';

  appendCoachMessage(firstQuestion, 0);

  const monitorPanel = $('monitor-panel');
  const chatBody     = $('chat-body');
  if (monitorNeeded) {
    monitorPanel.hidden = false;
    chatBody.classList.add('with-alignment');
    // Show/hide tabs based on config
    setTabVisible('tab-alignment', alignmentEnabled);
    setTabVisible('tab-certainty', S.wizard.uncertaintyTracking);
    setTabVisible('tab-meal', true);  // MealTracker always runs
    setTabVisible('tab-context', true);  // ContextTracker always runs (profile + summarization)
    setTabVisible('tab-dialogue-planner', true);  // Dialogue Planner always runs
    setTabVisible('tab-guardrail', true);  // Guardrail always runs
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
  if (mealCards) mealCards.innerHTML = '<div class="alignment-panel-empty" id="meal-panel-empty"><p>Meal Base will appear after MealTracker processes the conversation.</p></div>';
  const certCards = $('certainty-panel-cards');
  if (certCards) certCards.innerHTML = '<div class="alignment-panel-empty" id="certainty-panel-empty"><p>Certainty scores will appear when Uncertainty Tracking is enabled.</p></div>';
  const plannerCards = $('planner-panel-cards');
  if (plannerCards) plannerCards.innerHTML = '<div class="alignment-panel-empty" id="planner-panel-empty"><p>Dialogue Planner decisions and Meal Recommender results will appear here.</p></div>';
  const guardCards = $('guardrail-panel-cards');
  if (guardCards) guardCards.innerHTML = '<div class="alignment-panel-empty" id="guardrail-panel-empty"><p>Guardrail input/output will appear here for each turn.</p></div>';
  const ctxCards = $('context-panel-cards');
  if (ctxCards) ctxCards.innerHTML = '<div class="alignment-panel-empty" id="context-panel-empty"><p>Context Base will appear after ContextTracker processes the conversation.</p></div>';

  const inputArea = $('input-area');
  inputArea.hidden = false;
  const inp = $('user-input');
  inp.value    = '';
  inp.disabled = false;
  inp.style.height = '';
  $('send-btn').disabled = true;
  inp.focus();

  show('chat');
}


/* ═══════════════════════════════════════════════════════════════
   CUSTOM CHAT
═══════════════════════════════════════════════════════════════ */

/**
 * Create a coach message bubble and return the bubble element
 * for incremental text appending (streaming support).
 */
function appendCoachBubbleEmpty(turnIdx) {
  const row  = mkRow('coach');
  const av   = mkAvatar('Coach');
  const wrap = mkWrap();
  const hdr  = document.createElement('div');
  hdr.className = 'msg-header';
  hdr.appendChild(mkLabel(S.coachLabel || 'Coach'));
  if (turnIdx !== undefined && turnIdx !== null) hdr.appendChild(mkTurnBadge(turnIdx));
  const bub = mkBubble('');
  wrap.appendChild(hdr);
  wrap.appendChild(bub);
  row.appendChild(av);
  row.appendChild(wrap);
  $('messages').appendChild(row);
  scrollBottom();
  return bub;
}

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
    const res = await fetch('/api/session/' + S.sessionId + '/turn/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_reply: text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Server error ' + res.status);
    }

    // ── SSE 스트림 파싱 / Parse SSE stream ────────────────────────
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let meta = null;
    let assessmentText = null;
    let hadAssessment = false;
    let coachBub = null;
    let typingRemoved = false;
    let displayTurn = S.turnIdx + 1;  // coach 턴 = user 턴 + 1 (meta 에서 갱신)

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events (event: <type>\ndata: <payload>\n\n)
      const events = buffer.split('\n\n');
      buffer = events.pop(); // keep incomplete event in buffer

      for (const raw of events) {
        if (!raw.trim()) continue;
        const lines = raw.split('\n');
        let evType = '', evData = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) evType = line.slice(7);
          else if (line.startsWith('data: ')) evData = line.slice(6);
        }

        if (evType === 'meta') {
          meta = JSON.parse(evData);

          // Guardrail blocked
          if (meta.guardrail_blocked) {
            removeEl(typing); typingRemoved = true;
            // Still update monitoring so Guardrail tab shows the block details
            updateMonitorData(meta);
            const msg = meta.coach_question || meta.message || '';
            if (msg) appendCoachMessage('\u26a0\ufe0f ' + msg, S.turnIdx);
            $('user-input').disabled = false;
            $('send-btn').disabled   = true;
            $('user-input').focus();
            return;
          }

          // Status row + monitoring
          appendStatusRow(meta.turn_idx, meta.alignment_aligned, meta.alignment_score,
            meta.aligned_label, meta.certainty_score, meta.certainty_reasoning);
          if (S.wizard.alignmentEnabled) {
            appendAlignmentCard(meta.turn_idx, meta.alignment_aligned, meta.alignment_score,
              meta.aligned_label, meta.alignment_input, meta.alignment_raw_output, meta.alignment_reasoning);
          }
          updateMonitorData(meta);

        } else if (evType === 'assessment') {
          assessmentText = JSON.parse(evData);
          hadAssessment = true;

        } else if (evType === 'bubble_start') {
          // Multi-bubble: 새로운 말풍선 시작 (assessment 이후 후속 발화)
          if (!typingRemoved) { removeEl(typing); typingRemoved = true; }
          if (assessmentText) {
            appendCoachMessage(assessmentText, displayTurn);
            assessmentText = null;
          }
          coachBub = null;  // 새 말풍선 생성을 위해 리셋

        } else if (evType === 'token') {
          if (!typingRemoved) { removeEl(typing); typingRemoved = true; }
          // Assessment multi-bubble: show assessment before streaming follow-up
          if (assessmentText && !coachBub) {
            appendCoachMessage(assessmentText, displayTurn);
            assessmentText = null;
          }
          if (!coachBub) {
            coachBub = appendCoachBubbleEmpty(displayTurn);
          }
          const chunk = JSON.parse(evData);
          coachBub.textContent += chunk;
          scrollBottom();

        } else if (evType === 'done') {
          if (!typingRemoved) { removeEl(typing); typingRemoved = true; }
        }
      }
    }

    // Finalize
    if (!typingRemoved) removeEl(typing);

    if (meta && meta.status === 'active') {
      S.turnIdx = meta.turn_idx + 1;
      $('user-input').disabled = false;
      $('send-btn').disabled   = true;
      $('user-input').focus();
    } else if (meta) {
      S.ended = true;
      const data = meta;
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

    appendCoachMessage(data.first_question, 0);
    $('user-input').disabled = false;
    $('send-btn').disabled   = true;
    $('user-input').focus();
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

function appendSystemMessage(text) {
  const row = mkRow('system');
  const bub = mkBubble(text);
  row.appendChild(bub);
  $('messages').appendChild(row);
  scrollBottom();
}

function appendStatusRow(turnIdx, alignmentAligned, alignmentScore, alignedLabel, certaintyScore, certaintyReasoning) {
  // Deploy mode: no status chips visible to user
  if (S.mode === 'deploy') return;

  const hasAlign = S.wizard.alignmentEnabled && alignmentAligned !== null && alignmentAligned !== undefined;
  const hasCert  = certaintyScore !== null && certaintyScore !== undefined;
  if (!hasAlign && !hasCert) return;

  const row = document.createElement('div');
  row.className = 'message system status-row';

  if (hasAlign) {
    const scoreVal = (alignmentScore !== null && alignmentScore !== undefined) ? alignmentScore : (alignmentAligned ? 1.0 : 0.0);
    var aCls = scoreVal >= 0.7 ? 'aligned' : scoreVal >= 0.3 ? 'mid-aligned' : 'not-aligned';
    var aIcon = scoreVal >= 0.7 ? '🟢' : scoreVal >= 0.3 ? '🟡' : '🔴';
    const chip = document.createElement('div');
    chip.className = 'align-chip ' + aCls;
    chip.innerHTML = '<span class="align-chip-icon">' + aIcon + '</span>'
      + '<span>Alignment: ' + scoreVal.toFixed(2) + '</span>';
    row.appendChild(chip);
  }

  if (hasCert) {
    var cCls = certaintyScore >= 0.85 ? 'high' : certaintyScore >= 0.5 ? 'mid' : 'low';
    var cIcon = certaintyScore >= 0.85 ? '😃' : certaintyScore >= 0.5 ? '😐' : '😟';
    const chip = document.createElement('div');
    chip.className = 'certainty-chip ' + cCls;
    chip.innerHTML = '<span class="certainty-icon">' + cIcon + '</span>'
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
function closeModal() {
  $('confirm-modal').hidden = true;
  if (S._prevFocus) { S._prevFocus.focus(); S._prevFocus = null; }
}

function refreshAriaExpanded() {
  document.querySelectorAll('.alignment-card-toggle').forEach(toggle => {
    if (!toggle.hasAttribute('aria-expanded')) {
      const content = toggle.nextElementSibling;
      if (content) toggle.setAttribute('aria-expanded', String(!content.classList.contains('collapsed')));
    }
  });
}

/* Delegated aria-expanded update for collapsible toggle buttons */
document.addEventListener('click', (e) => {
  const toggle = e.target.closest('.alignment-card-toggle');
  if (!toggle) return;
  const content = toggle.nextElementSibling;
  if (content) {
    toggle.setAttribute('aria-expanded', String(!content.classList.contains('collapsed')));
  }
});


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
  /* ── Helper: collapsible section builder ───────────────────── */
  function _mkToggleSection(label, text, startOpen) {
    const section = document.createElement('div');
    section.className = 'alignment-card-section';
    const toggle = document.createElement('button');
    toggle.className = 'alignment-card-toggle';
    toggle.type = 'button';
    toggle.textContent = (startOpen ? '\u25BC' : '\u25B6') + ' ' + label;
    const pre = document.createElement('pre');
    pre.className = 'alignment-card-pre' + (startOpen ? '' : ' collapsed');
    pre.textContent = text;
    toggle.addEventListener('click', () => {
      const collapsed = pre.classList.toggle('collapsed');
      toggle.textContent = (collapsed ? '\u25B6' : '\u25BC') + ' ' + label;
    });
    section.appendChild(toggle);
    section.appendChild(pre);
    return section;
  }

  /* ── Helper: badge / chip builder ──────────────────────────── */
  function _mkBadge(text, cls) {
    const el = document.createElement('span');
    el.className = 'alignment-card-verdict ' + (cls || '');
    el.textContent = text;
    return el;
  }

  function _mkInfoRow(label, value, opts) {
    opts = opts || {};
    const row = document.createElement('div');
    row.className = 'monitor-info-row';
    const k = document.createElement('span');
    k.className = 'monitor-info-label';
    k.textContent = label;
    const v = document.createElement('span');
    v.className = 'monitor-info-value';
    if (opts.bold) v.style.fontWeight = '600';
    if (opts.italic) v.style.fontStyle = 'italic';
    if (opts.color) v.style.color = opts.color;
    v.textContent = value;
    row.appendChild(k);
    row.appendChild(v);
    return row;
  }

  function _appendCardToPanel(card, panelId, emptyId) {
    const empty = $(emptyId);
    if (empty) empty.hidden = true;
    const container = $(panelId);
    if (container) { container.appendChild(card); container.scrollTop = container.scrollHeight; }
  }

  /* ── Meal Tracker tab ──────────────────────────────────────── */
  if (data.meal_tracker_output) {
    const card = document.createElement('div');
    card.className = 'alignment-card';
    const header = document.createElement('div');
    header.className = 'alignment-card-header';
    const turnBadge = document.createElement('span');
    turnBadge.className = 'alignment-card-turn';
    turnBadge.textContent = 'Turn ' + data.turn_idx;
    header.appendChild(turnBadge);
    card.appendChild(header);

    if (data.meal_tracker_input) card.appendChild(_mkToggleSection('Input', data.meal_tracker_input, false));

    // Parse "- Key: value" lines into structured block
    const mtOut = data.meal_tracker_output;
    const mtLines = mtOut.split('\n').filter(function(l) { return l.trim(); });
    const mtParsed = [];
    mtLines.forEach(function(line) {
      const m = line.match(/^\s*-\s*([^:]+):\s*(.*)$/);
      if (m) mtParsed.push({ key: m[1].trim(), val: m[2].trim() });
    });
    if (mtParsed.length > 0) {
      const mtBlock = document.createElement('div');
      mtBlock.className = 'monitor-structured-block';
      const mtTitle = document.createElement('div');
      mtTitle.className = 'monitor-structured-title';
      mtTitle.textContent = 'Meal Base';
      mtBlock.appendChild(mtTitle);
      mtParsed.forEach(function(item) {
        const row = document.createElement('div');
        row.className = 'monitor-structured-row';
        const label = document.createElement('span');
        label.className = 'monitor-structured-label';
        label.textContent = item.key;
        const value = document.createElement('span');
        value.className = 'monitor-structured-value';
        value.textContent = item.val || '(none)';
        row.appendChild(label);
        row.appendChild(value);
        mtBlock.appendChild(row);
      });
      card.appendChild(mtBlock);
    } else {
      card.appendChild(_mkToggleSection('Output (Meal Base)', mtOut, true));
    }
    _appendCardToPanel(card, 'meal-panel-cards', 'meal-panel-empty');
  }

  /* ── Context Tracker tab ───────────────────────────────────── */
  if (data.context_tracker_output || data.context_tracker_input) {
    const card = document.createElement('div');
    card.className = 'alignment-card';
    const header = document.createElement('div');
    header.className = 'alignment-card-header';
    const turnBadge = document.createElement('span');
    turnBadge.className = 'alignment-card-turn';
    turnBadge.textContent = 'Turn ' + data.turn_idx;
    header.appendChild(turnBadge);
    if (!data.context_tracker_output) {
      header.appendChild(_mkBadge('Not updated this turn', 'not-aligned'));
    }
    card.appendChild(header);

    if (data.context_tracker_input) card.appendChild(_mkToggleSection('Input', data.context_tracker_input, false));

    // Parse [Section] blocks into structured key-value display
    const ctOut = data.context_tracker_output || '(ContextTracker did not run this turn)';
    const sectionRegex = /\[([^\]]+)\]\s*/g;
    const sections = [];
    let match;
    const indices = [];
    while ((match = sectionRegex.exec(ctOut)) !== null) {
      indices.push({ name: match[1], start: match.index, contentStart: match.index + match[0].length });
    }
    if (indices.length > 0) {
      for (let i = 0; i < indices.length; i++) {
        const end = (i + 1 < indices.length) ? indices[i + 1].start : ctOut.length;
        sections.push({ name: indices[i].name, body: ctOut.slice(indices[i].contentStart, end).trim() });
      }
      const ctBlock = document.createElement('div');
      ctBlock.className = 'monitor-structured-block';
      const ctTitle = document.createElement('div');
      ctTitle.className = 'monitor-structured-title';
      ctTitle.textContent = 'Context Base';
      ctBlock.appendChild(ctTitle);
      sections.forEach(function(sec) {
        const row = document.createElement('div');
        row.className = 'monitor-structured-row';
        const label = document.createElement('span');
        label.className = 'monitor-structured-label';
        label.textContent = sec.name;
        const value = document.createElement('span');
        value.className = 'monitor-structured-value';
        value.textContent = sec.body || '(none)';
        row.appendChild(label);
        row.appendChild(value);
        ctBlock.appendChild(row);
      });
      card.appendChild(ctBlock);
    } else {
      card.appendChild(_mkToggleSection('Output (Context Base)', ctOut, true));
    }
    _appendCardToPanel(card, 'context-panel-cards', 'context-panel-empty');
  }

  /* ── Certainty Tracker tab ─────────────────────────────────── */
  if (data.certainty_score !== null && data.certainty_score !== undefined) {
    const card = document.createElement('div');
    card.className = 'alignment-card';
    const header = document.createElement('div');
    header.className = 'alignment-card-header';
    const turnBadge = document.createElement('span');
    turnBadge.className = 'alignment-card-turn';
    turnBadge.textContent = 'Turn ' + data.turn_idx;
    header.appendChild(turnBadge);
    const cScore = data.certainty_score;
    var cCls = cScore >= 0.85 ? 'aligned' : cScore >= 0.5 ? 'mid-aligned' : 'not-aligned';
    var cIcon = cScore >= 0.85 ? '\uD83D\uDE03' : cScore >= 0.5 ? '\uD83D\uDE10' : '\uD83D\uDE1F';
    header.appendChild(_mkBadge(
      cIcon + ' Certainty: ' + cScore.toFixed(2),
      cCls
    ));
    card.appendChild(header);

    // Structured output block
    const cBlk = document.createElement('div');
    cBlk.className = 'monitor-structured-block';
    const cTitle = document.createElement('div');
    cTitle.className = 'monitor-structured-title';
    cTitle.textContent = 'Certainty Result';
    cBlk.appendChild(cTitle);
    var cColor = cScore >= 0.85 ? '#059669' : cScore >= 0.5 ? '#854d0e' : '#d97706';
    var cStatus = cScore >= 0.85 ? 'Sufficient information' : cScore >= 0.5 ? 'Partially sufficient' : 'More information needed';
    cBlk.appendChild(_mkInfoRow('Score', cScore.toFixed(2), { bold: true, color: cColor }));
    cBlk.appendChild(_mkInfoRow('Status', cStatus, { color: cColor }));
    if (data.certainty_reasoning) cBlk.appendChild(_mkInfoRow('Reasoning', data.certainty_reasoning));
    card.appendChild(cBlk);

    if (data.certainty_input) card.appendChild(_mkToggleSection('Input', data.certainty_input, false));
    if (data.certainty_output) card.appendChild(_mkToggleSection('Raw Output', data.certainty_output, false));
    _appendCardToPanel(card, 'certainty-panel-cards', 'certainty-panel-empty');
  }

  /* ── Dialogue Planner tab — structured decision card ────────────── */
  const plannerDecision = data.dialogue_plan;
  const plannerRawOutput = data.dialogue_planner_raw_output;
  const plannerInput = data.dialogue_planner_input;
  if (plannerDecision) {
    const card = document.createElement('div');
    card.className = 'alignment-card planner-card';
    const dec = plannerDecision;

    const phaseLabels = {
      'exploration':'Exploration','assessment':'Assessment',
      'recommendation':'Recommendation','negotiation':'Negotiation',
      'confirmation':'Confirmation','finalization':'Finalization',
      'motivational_ending':'Finalization','terminated':'Terminated',
      'info_seeking':'Info Seeking','rec_info_seeking':'Rec Info Seeking',
      'recommending':'Recommending',
    };
    const actionLabels = {
      'inquire':'Inquire','assess':'Assess','terminate':'Terminate','recommend':'Recommend',
      'evaluate':'Evaluate','seek_rec_info':'Seek Rec Info',
      'seek_meal_info':'Seek Meal Info','motivational_close':'Close',
      'respond':'Respond','confirm':'Confirm','handoff':'Handoff','close':'Close',
    };
    const phaseActions = {
      'exploration':       ['inquire','respond','assess','terminate'],
      'assessment':        ['assess','terminate'],
      'recommendation':    ['recommend','respond','confirm','terminate'],
      'negotiation':       ['respond','inquire','assess','recommend','handoff','confirm','close','terminate'],
      'confirmation':      ['confirm','respond','assess','close','terminate'],
      'finalization':      ['close','terminate'],
      'info_seeking':       ['seek_meal_info','evaluate','respond','terminate'],
      'rec_info_seeking':   ['seek_rec_info','recommend','respond','terminate'],
      'recommending':       ['assess','recommend','confirm','respond','terminate'],
      'motivational_ending':['close','terminate'],
      'terminated':         ['terminate'],
    };
    const allIntents = ['informing','accepting','inquiring','deferring','passive','rejecting','disengaging'];

    // helper: chip-selector row (like rec-type)
    function _mkChipRow(label, allOptions, activeValue, labelMap) {
      const row = document.createElement('div');
      row.className = 'monitor-info-row';
      row.style.padding = '8px 12px';
      row.style.borderBottom = '1px solid #f1f5f9';
      const k = document.createElement('span');
      k.className = 'monitor-info-label';
      k.textContent = label;
      row.appendChild(k);
      const chips = document.createElement('span');
      chips.className = 'rec-type-chips';
      var active = (activeValue || '').toLowerCase();
      allOptions.forEach(function(opt) {
        const chip = document.createElement('span');
        chip.className = 'rec-type-chip' + (opt === active ? ' active' : '');
        chip.textContent = (labelMap && labelMap[opt]) || opt;
        chips.appendChild(chip);
      });
      row.appendChild(chips);
      return row;
    }

    // ── Header: Turn badge only
    const header = document.createElement('div');
    header.className = 'alignment-card-header';
    const turnBadge = document.createElement('span');
    turnBadge.className = 'alignment-card-turn';
    turnBadge.textContent = 'Turn ' + data.turn_idx;
    header.appendChild(turnBadge);
    card.appendChild(header);

    // ── Routing Decision block
    const decBlock = document.createElement('div');
    decBlock.className = 'monitor-structured-block orch-info';
    const decTitle = document.createElement('div');
    decTitle.className = 'monitor-structured-title';
    decTitle.textContent = 'Routing Decision';
    decBlock.appendChild(decTitle);

    // 1) Selected Action — chip selector (phase-specific actions)
    var currentPhaseActions = phaseActions[data.phase] || ['terminate'];
    decBlock.appendChild(_mkChipRow('Selected Action', currentPhaseActions, dec.action, actionLabels));
    // 2) Current Phase — plain text info row
    var phaseRow = document.createElement('div');
    phaseRow.className = 'monitor-info-row';
    phaseRow.style.padding = '8px 12px';
    phaseRow.style.borderBottom = '1px solid #f1f5f9';
    var phaseLabel = document.createElement('span');
    phaseLabel.className = 'monitor-info-label';
    phaseLabel.textContent = 'Current Phase';
    phaseRow.appendChild(phaseLabel);
    var phaseValue = document.createElement('span');
    phaseValue.className = 'monitor-info-value';
    phaseValue.style.fontWeight = '600';
    phaseValue.textContent = (phaseLabels[data.phase] || data.phase);
    phaseRow.appendChild(phaseValue);
    decBlock.appendChild(phaseRow);
    // 3) User Intent — chip selector
    if (dec.user_intent) {
      decBlock.appendChild(_mkChipRow('User Intent', allIntents, dec.user_intent, null));
    }
    // remove last row border
    var dLast = decBlock.lastElementChild;
    if (dLast) dLast.style.borderBottom = 'none';
    card.appendChild(decBlock);

    // ── Intent Summary (quote block)
    if (dec.intent_summary) {
      const intentBlock = document.createElement('div');
      intentBlock.className = 'orch-intent';
      const intentLabel = document.createElement('div');
      intentLabel.className = 'orch-section-label';
      intentLabel.textContent = 'Intent Summary';
      intentBlock.appendChild(intentLabel);
      const intentBody = document.createElement('div');
      intentBody.className = 'orch-intent-body';
      intentBody.textContent = dec.intent_summary;
      intentBlock.appendChild(intentBody);
      card.appendChild(intentBlock);
    }

    // ── Routing Reasoning (collapsible, closed by default)
    if (dec.reasoning) {
      card.appendChild(_mkToggleSection('Routing Reasoning', dec.reasoning, false));
    }
    // ── Internal Guidance To Sub-Agent (collapsible, closed)
    if (dec.instruction) {
      card.appendChild(_mkToggleSection('Internal Guidance To Sub-Agent', dec.instruction, false));
    }

    // ── Assessment result
    if (data.assessment_result) {
      const assess = data.assessment_result;
      const aBlk = document.createElement('div');
      aBlk.className = 'monitor-structured-block';
      const aTitle = document.createElement('div');
      aTitle.className = 'monitor-structured-title';
      aTitle.textContent = '\uD83D\uDCCB Assessment';
      aBlk.appendChild(aTitle);
      if (assess.overall) aBlk.appendChild(_mkInfoRow('Overall', assess.overall, { bold: true }));
      if (assess.strengths) aBlk.appendChild(_mkInfoRow('Strengths', assess.strengths));
      if (assess.gaps) aBlk.appendChild(_mkInfoRow('Gaps', assess.gaps));
      if (assess.suggestion) aBlk.appendChild(_mkInfoRow('Suggestion', assess.suggestion));
      card.appendChild(aBlk);
    }

    // ── Recommendation result
    if (data.recommendation_result) {
      const rec = data.recommendation_result;
      const rBlk = document.createElement('div');
      rBlk.className = 'monitor-structured-block';
      const rTitle = document.createElement('div');
      rTitle.className = 'monitor-structured-title';
      rTitle.textContent = '\uD83D\uDCA1 Recommendation';
      rBlk.appendChild(rTitle);

      // Type badge selector: show all types, highlight active
      const typeRow = document.createElement('div');
      typeRow.className = 'monitor-info-row';
      typeRow.style.padding = '8px 12px';
      typeRow.style.borderBottom = '1px solid #f1f5f9';
      const typeLabel = document.createElement('span');
      typeLabel.className = 'monitor-info-label';
      typeLabel.textContent = 'Type';
      typeRow.appendChild(typeLabel);
      const typeChips = document.createElement('span');
      typeChips.className = 'rec-type-chips';
      var activeType = (rec.recommendation_type || '').toLowerCase();
      ['add', 'modify', 'substitute'].forEach(function(t) {
        const chip = document.createElement('span');
        chip.className = 'rec-type-chip' + (t === activeType ? ' active' : '');
        chip.textContent = t;
        typeChips.appendChild(chip);
      });
      typeRow.appendChild(typeChips);
      rBlk.appendChild(typeRow);

      if (rec.target_food) rBlk.appendChild(_mkInfoRow('Target', rec.target_food));
      if (rec.suggestion) rBlk.appendChild(_mkInfoRow('Suggestion', rec.suggestion));
      if (rec.reasoning) rBlk.appendChild(_mkInfoRow('Reasoning', rec.reasoning));
      if (rec.expected_impact) rBlk.appendChild(_mkInfoRow('Expected Impact', rec.expected_impact));
      card.appendChild(rBlk);
    }

    // ── Raw I/O collapsibles
    if (plannerInput) card.appendChild(_mkToggleSection('Input (Planner Prompt)', plannerInput, false));
    if (plannerRawOutput !== null && plannerRawOutput !== undefined) {
      card.appendChild(_mkToggleSection('Raw Output', plannerRawOutput || '(empty)', false));
    }
    _appendCardToPanel(card, 'planner-panel-cards', 'planner-panel-empty');
  }

  /* ── Guardrail tab — Input Guard + Output Guard structured ──── */
  const hasInputGuard  = (data.input_guard_output !== null && data.input_guard_output !== undefined);
  const hasOutputGuard = (data.output_guard_output !== null && data.output_guard_output !== undefined);

  if (hasInputGuard || hasOutputGuard) {
    const card = document.createElement('div');
    card.className = 'alignment-card';

    const header = document.createElement('div');
    header.className = 'alignment-card-header';
    const turnBadge = document.createElement('span');
    turnBadge.className = 'alignment-card-turn';
    turnBadge.textContent = 'Turn ' + data.turn_idx;
    header.appendChild(turnBadge);

    const blocked = data.guardrail_blocked;
    let ogFailed = false;
    if (hasOutputGuard) {
      try {
        const _ogCheck = JSON.parse(data.output_guard_output);
        if (_ogCheck.passed === false) ogFailed = true;
      } catch(e) {}
    }
    const guardStatus = blocked ? '\u26D4 Blocked' : ogFailed ? '\u26A0\uFE0F OG Flagged' : '\u2705 Passed';
    const guardCls    = blocked ? 'not-aligned' : ogFailed ? 'not-aligned' : 'aligned';
    header.appendChild(_mkBadge(guardStatus, guardCls));
    card.appendChild(header);

    // ── Input Guard — structured block
    if (hasInputGuard) {
      const igBlk = document.createElement('div');
      igBlk.className = 'monitor-structured-block';
      const igTitle = document.createElement('div');
      igTitle.className = 'monitor-structured-title';
      igTitle.textContent = '\uD83D\uDEE1\uFE0F Input Guard';
      igBlk.appendChild(igTitle);

      try {
        const igParsed = JSON.parse(data.input_guard_output);
        var igAction = igParsed.action || 'pass';
        var igActionColor = igAction === 'pass' ? '#059669' : '#b91c1c';
        igBlk.appendChild(_mkInfoRow('Result', igAction === 'pass' ? '\u2705 Passed' : '\u274C ' + igAction, { bold: true, color: igActionColor }));
        if (igParsed.flags && igParsed.flags.length > 0) {
          igBlk.appendChild(_mkInfoRow('Flags', igParsed.flags.join(', ')));
        }
        if (igParsed.reason) {
          igBlk.appendChild(_mkInfoRow('Reason', igParsed.reason));
        }
      } catch(e) {
        igBlk.appendChild(_mkInfoRow('Result', 'Parse error', { color: '#94a3b8' }));
      }
      card.appendChild(igBlk);
      if (data.input_guard_input) card.appendChild(_mkToggleSection('Input Guard Input', data.input_guard_input, false));
      card.appendChild(_mkToggleSection('Input Guard Raw Output', data.input_guard_output, false));
    }

    // ── Output Guard — structured block
    if (hasOutputGuard) {
      const ogBlk = document.createElement('div');
      ogBlk.className = 'monitor-structured-block';
      const ogTitle = document.createElement('div');
      ogTitle.className = 'monitor-structured-title';
      ogTitle.textContent = '\uD83D\uDD12 Output Guard';
      ogBlk.appendChild(ogTitle);

      try {
        const ogParsed = JSON.parse(data.output_guard_output);
        const ogPassed = ogParsed.passed !== false;
        ogBlk.appendChild(_mkInfoRow('Result', ogPassed ? '\u2705 Passed' : '\u274C Flagged', { bold: true, color: ogPassed ? '#059669' : '#b91c1c' }));
        if (ogParsed.reason) {
          ogBlk.appendChild(_mkInfoRow('Reason', ogParsed.reason));
        }
        if (ogPassed && !ogParsed.reason) {
          ogBlk.appendChild(_mkInfoRow('Detail', 'No issues detected in coach response.', { color: '#059669' }));
        }
      } catch(e) {
        ogBlk.appendChild(_mkInfoRow('Result', 'Parse error', { color: '#94a3b8' }));
      }
      card.appendChild(ogBlk);
      if (data.output_guard_input) card.appendChild(_mkToggleSection('Output Guard Input', data.output_guard_input, false));
      card.appendChild(_mkToggleSection('Output Guard Raw Output', data.output_guard_output, false));
    }

    _appendCardToPanel(card, 'guardrail-panel-cards', 'guardrail-panel-empty');
  }

  refreshAriaExpanded();
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
                : (function(){ var s = (alignmentScore !== null && alignmentScore !== undefined) ? alignmentScore : (alignmentAligned ? 1.0 : 0.0); return s >= 0.7 ? 'aligned' : s >= 0.3 ? 'mid-aligned' : 'not-aligned'; })()
                ;
  const verdict = document.createElement('span');
  verdict.className = 'alignment-card-verdict ' + chipCls;
  if (alignmentAligned === null) {
    verdict.textContent = 'Pending';
  } else {
    const scoreVal = (alignmentScore !== null && alignmentScore !== undefined) ? alignmentScore : (alignmentAligned ? 1.0 : 0.0);
    var aIcon = scoreVal >= 0.7 ? '\uD83D\uDFE2' : scoreVal >= 0.3 ? '\uD83D\uDFE1' : '\uD83D\uDD34';
    verdict.textContent = aIcon + ' Alignment: ' + scoreVal.toFixed(2);
  }
  header.appendChild(verdict);
  card.appendChild(header);

  /* Structured output block */
  if (alignmentAligned !== null) {
    const scoreVal = (alignmentScore !== null && alignmentScore !== undefined) ? alignmentScore : (alignmentAligned ? 1.0 : 0.0);
    var aColor = scoreVal >= 0.7 ? '#059669' : scoreVal >= 0.3 ? '#854d0e' : '#b91c1c';
    const aBlk = document.createElement('div');
    aBlk.className = 'monitor-structured-block';
    const aTitle = document.createElement('div');
    aTitle.className = 'monitor-structured-title';
    aTitle.textContent = 'Alignment Result';
    aBlk.appendChild(aTitle);

    // Score info row
    const scoreRow = document.createElement('div');
    scoreRow.className = 'monitor-info-row';
    scoreRow.style.padding = '6px 12px';
    scoreRow.style.borderBottom = '1px solid #f1f5f9';
    const scoreLabel = document.createElement('span');
    scoreLabel.className = 'monitor-info-label';
    scoreLabel.textContent = 'Score';
    scoreRow.appendChild(scoreLabel);
    const scoreValue = document.createElement('span');
    scoreValue.className = 'monitor-info-value';
    scoreValue.style.fontWeight = '600';
    scoreValue.style.color = aColor;
    scoreValue.textContent = scoreVal.toFixed(2);
    scoreRow.appendChild(scoreValue);
    aBlk.appendChild(scoreRow);

    // Status info row
    const statusRow = document.createElement('div');
    statusRow.className = 'monitor-info-row';
    statusRow.style.padding = '6px 12px';
    statusRow.style.borderBottom = '1px solid #f1f5f9';
    const statusLabel = document.createElement('span');
    statusLabel.className = 'monitor-info-label';
    statusLabel.textContent = 'Status';
    statusRow.appendChild(statusLabel);
    const statusValue = document.createElement('span');
    statusValue.className = 'monitor-info-value';
    statusValue.style.fontWeight = '600';
    statusValue.style.color = aColor;
    statusValue.textContent = scoreVal >= 0.7 ? 'Aligned' : scoreVal >= 0.3 ? 'Partially Aligned' : 'Not Aligned';
    statusRow.appendChild(statusValue);
    aBlk.appendChild(statusRow);

    // Reasoning
    if (alignmentReasoning) {
      const reasonRow = document.createElement('div');
      reasonRow.className = 'monitor-info-row';
      reasonRow.style.padding = '6px 12px';
      const reasonLabel = document.createElement('span');
      reasonLabel.className = 'monitor-info-label';
      reasonLabel.textContent = 'Reasoning';
      reasonRow.appendChild(reasonLabel);
      const reasonValue = document.createElement('span');
      reasonValue.className = 'monitor-info-value';
      reasonValue.textContent = alignmentReasoning;
      reasonRow.appendChild(reasonValue);
      aBlk.appendChild(reasonRow);
    }
    card.appendChild(aBlk);
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
  refreshAriaExpanded();
}
