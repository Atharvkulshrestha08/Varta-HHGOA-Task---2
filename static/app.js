/**
 * VoiceRAG — Frontend Application Logic
 *
 * Handles:
 * 1. Microphone capture & voice recording (MediaRecorder API)
 * 2. Waveform visualization (Web Audio API + Canvas)
 * 3. Voice & text query submission
 * 4. Answer rendering with source citations
 * 5. Latency analytics dashboard
 * 6. Health check polling
 */

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════

const state = {
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    audioContext: null,
    analyser: null,
    animationFrame: null,
    selectedLanguage: 'auto',
    conversationHistory: [],
    sessionId: 'sess_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7),
};

// ═══════════════════════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════════════════════

const $ = (id) => document.getElementById(id);

const els = {
    micButton: $('mic-button'),
    micLabel: $('mic-label'),
    waveformCanvas: $('waveform-canvas'),
    textForm: $('text-form'),
    textInput: $('text-input'),
    sendBtn: $('send-btn'),
    statusBar: $('status-bar'),
    statusText: $('status-text'),
    emptyState: $('empty-state'),
    answerContent: $('answer-content'),
    queryDisplay: $('query-display'),
    answerText: $('answer-text'),
    answerCard: $('answer-card'),
    confidenceDisplay: $('confidence-display'),
    languageDisplay: $('language-display'),
    guardrailFlags: $('guardrail-flags'),
    sourcesList: $('sources-list'),
    latencyBars: $('latency-bars'),
    totalLatency: $('total-latency'),
    healthBadge: $('health-badge'),
    vectorsBadge: $('vectors-badge'),
    refreshAnalytics: $('refresh-analytics'),
    analyticsGrid: $('analytics-grid'),
    stageBreakdown: $('stage-breakdown'),
    toggleTeachBtn: $('toggle-teach-btn'),
    teachDrawer: $('teach-drawer'),
    teachDrawerClose: $('teach-drawer-close'),
    teachForm: $('teach-form'),
    teachInput: $('teach-input'),
    teachSubmitBtn: $('teach-submit-btn'),
    teachStatus: $('teach-status'),
    chatThreadWrapper: $('chat-thread-wrapper'),
    chatThreadMessages: $('chat-thread-messages'),
    resetThreadBtn: $('reset-thread-btn'),
};

// ═══════════════════════════════════════════════════════════════
// Language Selection
// ═══════════════════════════════════════════════════════════════

document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.selectedLanguage = btn.dataset.lang;
    });
});

// ═══════════════════════════════════════════════════════════════
// Voice Recording
// ═══════════════════════════════════════════════════════════════

let lastToggleTime = 0;

if (els.micButton) {
    els.micButton.addEventListener('click', toggleRecording);
}

async function toggleRecording(e) {
    if (e && e.preventDefault) e.preventDefault();
    if (e && e.stopPropagation) e.stopPropagation();

    const now = Date.now();
    if (now - lastToggleTime < 400) {
        console.log('[MIC] Debounced quick double click.');
        return;
    }
    lastToggleTime = now;

    console.log('[MIC] toggleRecording triggered. Current isRecording =', state.isRecording);
    if (state.isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Microphone access is only available on localhost or HTTPS. Please use Chrome/Edge or use the text input below.');
        return;
    }

    try {
        els.micLabel.textContent = 'Starting mic...';

        // Pre-warm TCP/HTTP connection in background for 0-latency POST query submission
        fetch('/api/health', { method: 'GET', cache: 'no-store' }).catch(() => {});

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        // Setup audio context for waveform & VAD silence detection
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (state.audioContext.state === 'suspended') {
            await state.audioContext.resume();
        }

        state.analyser = state.audioContext.createAnalyser();
        state.analyser.fftSize = 256;
        const source = state.audioContext.createMediaStreamSource(stream);
        source.connect(state.analyser);

        // Start MediaRecorder
        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm',
        });

        state.audioChunks = [];
        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
                state.audioChunks.push(e.data);
            }
        };

        state.mediaRecorder.onstop = () => {
            console.log('[MIC] MediaRecorder stopped. Total chunks collected:', state.audioChunks.length);
            const blob = new Blob(state.audioChunks, { type: state.mediaRecorder.mimeType });
            stream.getTracks().forEach(t => t.stop());
            if (state.silenceTimer) clearInterval(state.silenceTimer);
            if (state.maxDurationTimer) clearTimeout(state.maxDurationTimer);
            submitVoiceQuery(blob);
        };

        state.mediaRecorder.start(250); // Emit chunk every 250ms
        state.isRecording = true;
        state.hasSpoken = false;
        state.silenceStart = null;
        state.recordStartTime = Date.now();

        els.micButton.classList.add('recording');
        els.micLabel.textContent = 'Listening... (Tap to stop)';
        drawWaveform();

        // Safety Max Timer: auto-stops after 8 seconds max
        state.maxDurationTimer = setTimeout(() => {
            if (state.isRecording) {
                console.log('[MIC] Reached 8s max duration. Auto-stopping...');
                els.micLabel.textContent = 'Submitting...';
                stopRecording();
            }
        }, 8000);

        // VAD Silence Detection: auto-stops 1.8s after user stops speaking
        const pBuffer = new Uint8Array(state.analyser.frequencyBinCount);
        state.silenceTimer = setInterval(() => {
            if (!state.isRecording || !state.analyser) return;

            state.analyser.getByteFrequencyData(pBuffer);
            let sSum = 0;
            for (let i = 0; i < pBuffer.length; i++) sSum += pBuffer[i];
            const avgVal = sSum / pBuffer.length;

            if (avgVal > 8) {
                // User is actively speaking
                state.hasSpoken = true;
                state.silenceStart = null;
                els.micLabel.textContent = 'Listening... (Tap to stop)';
            } else if (state.hasSpoken) {
                // User was speaking and is now silent
                if (!state.silenceStart) {
                    state.silenceStart = Date.now();
                } else if (Date.now() - state.silenceStart >= 1800) {
                    console.log('[VAD] Silence detected for 1.8s. Auto-stopping recording...');
                    els.micLabel.textContent = 'Processing speech...';
                    stopRecording();
                }
            }
        }, 100);

    } catch (err) {
        console.error('Microphone access error:', err);
        state.isRecording = false;
        els.micButton.classList.remove('recording');
        els.micLabel.textContent = 'Mic Error';
        alert('Microphone permission was denied. Please allow microphone access in your browser address bar and try again.');
    }
}

function stopRecording() {
    console.log('[MIC] stopRecording invoked. Current state:', state.isRecording);
    if (!state.isRecording) return;

    state.isRecording = false;
    els.micButton.classList.remove('recording');
    els.micLabel.textContent = 'Processing...';
    cancelAnimationFrame(state.animationFrame);

    if (state.silenceTimer) clearInterval(state.silenceTimer);
    if (state.maxDurationTimer) clearTimeout(state.maxDurationTimer);

    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }

    if (state.audioContext) {
        try { state.audioContext.close(); } catch (e) {}
        state.audioContext = null;
    }
}

// ═══════════════════════════════════════════════════════════════
// Waveform Visualization
// ═══════════════════════════════════════════════════════════════

function drawWaveform() {
    const canvas = els.waveformCanvas;
    const ctx = canvas.getContext('2d');
    const analyser = state.analyser;
    if (!analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        if (!state.isRecording) return;
        state.animationFrame = requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        // Clear
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw waveform
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#10b981'; // HH Goa brand green
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * canvas.height) / 2;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();

        // Glow effect
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.3)'; // HH Goa brand pink glow
        ctx.lineWidth = 4;
        ctx.stroke();
    }

    draw();
}

// Draw idle waveform
function drawIdleWaveform() {
    const canvas = els.waveformCanvas;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.2)'; // Brand green
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

drawIdleWaveform();

// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════
// Query Submission
// ═══════════════════════════════════════════════════════════════

// Text form submission
els.textForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = els.textInput.value.trim();
    if (!text) return;
    submitTextQuery(text);
    els.textInput.value = '';
});

async function submitVoiceQuery(audioBlob) {
    showStatus('Transcribing speech...');

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    if (state.selectedLanguage !== 'auto') {
        formData.append('language_hint', state.selectedLanguage);
    }
    formData.append('top_k', '5');
    formData.append('session_id', state.sessionId);
    formData.append('conversation_history', JSON.stringify(state.conversationHistory));

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        renderAnswer(data);
    } catch (err) {
        console.error('Voice query error:', err);
        showError('Failed to process voice query. Please try again or use text input.');
    } finally {
        hideStatus();
    }
}

async function submitTextQuery(text) {
    showStatus('Searching & generating answer...');

    try {
        const body = {
            text: text,
            top_k: 5,
            session_id: state.sessionId,
            conversation_history: state.conversationHistory,
        };
        if (state.selectedLanguage !== 'auto') {
            body.language_hint = state.selectedLanguage;
        }

        const response = await fetch('/api/query/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const data = await response.json();
        renderAnswer(data);
    } catch (err) {
        console.error('Text query error:', err);
        showError('Failed to process query. Please try again.');
    } finally {
        hideStatus();
    }
}

// ═══════════════════════════════════════════════════════════════
// Answer Rendering
// ═══════════════════════════════════════════════════════════════

const LANG_NAMES = {
    'hin_Deva': 'Hindi',
    'ben_Beng': 'Bengali',
    'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu',
    'eng_Latn': 'English',
    'unknown': 'Auto-detected',
};

function renderAnswer(data) {
    // Hide empty state, show content
    els.emptyState.style.display = 'none';
    els.answerContent.style.display = 'block';

    // Query echo
    els.queryDisplay.textContent = data.original_query || 'Voice query';

    // Answer text (formatted with transliteration & translation support)
    els.answerText.innerHTML = renderFormattedAnswer(data.answer || 'No answer generated.');

    // Add to multi-turn conversation history
    if (data.original_query && data.answer) {
        state.conversationHistory.push({
            role: 'user',
            text: data.original_query,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        });
        state.conversationHistory.push({
            role: 'assistant',
            text: data.answer,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        });
        renderChatThread();
    }

    // Confidence
    const confidence = data.confidence || 0;
    els.confidenceDisplay.textContent = `${(confidence * 100).toFixed(0)}% confidence`;
    els.confidenceDisplay.style.color = confidence > 0.5 ? '#22c55e' : confidence > 0.3 ? '#f59e0b' : '#ef4444';

    // Language
    const lang = LANG_NAMES[data.language] || data.language || 'Unknown';
    els.languageDisplay.textContent = lang;

    // Guardrail flags
    if (data.guardrail_flags && data.guardrail_flags.length > 0) {
        els.guardrailFlags.style.display = 'block';
        els.guardrailFlags.innerHTML = data.guardrail_flags.map(f =>
            `<div class="guardrail-flag ${f.severity || 'warning'}">
                <span>${f.severity === 'block' ? '🛑' : '⚠️'}</span>
                <span>${f.reason}</span>
            </div>`
        ).join('');
    } else {
        els.guardrailFlags.style.display = 'none';
    }

    // Sources
    if (data.sources && data.sources.length > 0) {
        els.sourcesList.innerHTML = data.sources.map(s => {
            const isWiki = s.strategy === 'wikipedia_retrieval' || (s.source && s.source.toLowerCase().includes('wikipedia'));
            return `
            <div class="source-card ${isWiki ? 'source-card-wiki' : ''}">
                <div class="source-header">
                    <span class="source-rank">${isWiki ? '📖 Wikipedia Intelligence' : `Passage ${s.rank + 1}`}</span>
                    <span>
                        <span class="source-score">Relevance: ${s.score.toFixed(2)}</span>
                        ${s.strategy ? `<span class="source-strategy ${isWiki ? 'strategy-wiki' : ''}">${isWiki ? '🌐 Live Wiki Grounding' : s.strategy}</span>` : ''}
                    </span>
                </div>
                <div class="source-body">${escapeHtml(s.text)}</div>
                ${s.url ? `<div class="source-wiki-link" style="margin-top: 8px; font-size: 0.82rem;"><a href="${encodeURI(s.url)}" target="_blank" rel="noopener noreferrer" style="color: #38bdf8; text-decoration: underline; display: inline-flex; align-items: center; gap: 4px; font-weight: 500;">🔗 Read Full Article on Wikipedia &rarr;</a></div>` : ''}
            </div>`;
        }).join('');
    } else {
        els.sourcesList.innerHTML = `
            <div style="background: rgba(245, 213, 32, 0.08); border: 1px dashed rgba(245, 213, 32, 0.4); border-radius: 8px; padding: 14px; color: var(--text-secondary); font-size: 0.85rem; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.2rem;">💡</span>
                <span><strong>Grounded Response:</strong> Synthesized across conversational dialogue context, live Wikipedia intelligence, and Vartaलाप's active vector store.</span>
            </div>
        `;
    }

    // Latency breakdown
    if (data.latency_ms && Object.keys(data.latency_ms).length > 0) {
        const maxLatency = Math.max(...Object.values(data.latency_ms), 1);
        els.latencyBars.innerHTML = Object.entries(data.latency_ms).map(([stage, ms]) =>
            `<div class="latency-bar">
                <span class="latency-bar-label">${stage}</span>
                <div class="latency-bar-track">
                    <div class="latency-bar-fill" style="width: ${(ms / maxLatency * 100).toFixed(1)}%"></div>
                </div>
                <span class="latency-bar-value">${ms.toFixed(1)} ms</span>
            </div>`
        ).join('');

        const totalMs = data.total_latency_ms || 0;
        if (totalMs < 200 && totalMs > 0) {
            els.totalLatency.innerHTML = `${totalMs.toFixed(1)} ms <span style="color: #22c55e; font-size: 0.8rem; font-weight: 700; margin-left: 6px;">⚡ (&lt;200ms SLA MET)</span>`;
        } else {
            els.totalLatency.textContent = `${totalMs.toFixed(1)} ms`;
        }
    }

    // Setup HITL Feedback buttons for this query
    if (typeof setupHITLFeedback === 'function') {
        setupHITLFeedback(data.query_id || ('q_' + Date.now()), data.original_query || '', data.answer || '');
    }

    // Refresh analytics
    fetchAnalytics();

    // Scroll to answer
    els.answerCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderChatThread() {
    if (!els.chatThreadWrapper || !els.chatThreadMessages) return;
    if (state.conversationHistory.length <= 2) {
        els.chatThreadWrapper.style.display = 'none';
        return;
    }
    els.chatThreadWrapper.style.display = 'block';

    // Render previous turns (prior to latest turn)
    const prevTurns = state.conversationHistory.slice(0, -2);
    els.chatThreadMessages.innerHTML = prevTurns.map(item => {
        const isUser = item.role === 'user';
        const cleanContent = isUser ? escapeHtml(item.text) : renderFormattedAnswer(item.text);
        return `
            <div class="chat-bubble-row ${isUser ? 'user-row' : 'assistant-row'}">
                <div class="chat-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}">
                    <div class="chat-bubble-header">
                        <span class="chat-role">${isUser ? '👤 You' : '🤖 Vartaलाप'}</span>
                        <span class="chat-time">${item.timestamp || ''}</span>
                    </div>
                    <div class="chat-bubble-body">${cleanContent}</div>
                </div>
            </div>
        `;
    }).join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderFormattedAnswer(text) {
    if (!text) return 'No answer generated.';
    let escaped = escapeHtml(text);
    
    // Convert bold **text** to <strong>text</strong>
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Convert italic *text* or _text_ to <em>text</em>
    escaped = escaped.replace(/\*([^\*]+)\*/g, '<em>$1</em>');
    // Convert source citation pill [Source: ...]
    escaped = escaped.replace(/\[Source:\s*([^\]]+)\]/g, '<span class="source-inline-badge">📎 Source: $1</span>');
    
    return escaped;
}

// ═══════════════════════════════════════════════════════════════
// Status & Error Helpers
// ═══════════════════════════════════════════════════════════════

function showStatus(text) {
    els.statusBar.style.display = 'flex';
    els.statusText.textContent = text;
}

function hideStatus() {
    els.statusBar.style.display = 'none';
}

function showError(message) {
    els.emptyState.style.display = 'none';
    els.answerContent.style.display = 'block';
    els.queryDisplay.textContent = 'Error';
    els.answerText.textContent = message;
    els.confidenceDisplay.textContent = '';
    els.languageDisplay.textContent = '';
    els.guardrailFlags.style.display = 'none';
    els.sourcesList.innerHTML = '';
    els.latencyBars.innerHTML = '';
    els.totalLatency.textContent = '—';
}

// ═══════════════════════════════════════════════════════════════
// Analytics Dashboard
// ═══════════════════════════════════════════════════════════════

els.refreshAnalytics.addEventListener('click', fetchAnalytics);

async function fetchAnalytics() {
    try {
        const response = await fetch('/api/analytics');
        const data = await response.json();
        renderAnalytics(data);
    } catch (err) {
        console.error('Analytics fetch error:', err);
    }
}

function renderAnalytics(data) {
    const stats = data.statistics || {};
    const pipeline = stats.total_pipeline || {};

    $('val-p50').textContent = pipeline.p50 != null ? pipeline.p50.toFixed(1) : '—';
    $('val-p70').textContent = pipeline.p70 != null ? pipeline.p70.toFixed(1) : '—';
    $('val-p100').textContent = pipeline.p100 != null ? pipeline.p100.toFixed(1) : '—';
    $('val-queries').textContent = stats.total_queries || 0;

    // Color P50 based on target
    const p50El = $('val-p50');
    if (pipeline.p50 != null) {
        if (pipeline.p50 < 200) {
            p50El.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
        } else {
            p50El.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
        }
        p50El.style.webkitBackgroundClip = 'text';
        p50El.style.webkitTextFillColor = 'transparent';
    }

    // Stage breakdown
    const perStage = stats.per_stage || {};
    els.stageBreakdown.innerHTML = Object.entries(perStage).map(([stage, s]) =>
        `<div class="stage-card">
            <span class="stage-name">${stage}</span>
            <span class="stage-value">P50: ${s.p50.toFixed(1)}ms</span>
        </div>`
    ).join('');
}

// ═══════════════════════════════════════════════════════════════
// Health Check
// ═══════════════════════════════════════════════════════════════

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        if (data.status === 'healthy') {
            els.healthBadge.textContent = '● Online';
            els.healthBadge.style.color = '#22c55e';
        } else {
            els.healthBadge.textContent = '● Degraded';
            els.healthBadge.style.color = '#f59e0b';
        }

        els.vectorsBadge.textContent = `${(data.total_vectors || 0).toLocaleString()} vectors`;
    } catch (err) {
        els.healthBadge.textContent = '● Offline';
        els.healthBadge.style.color = '#ef4444';
    }
}

// ═══════════════════════════════════════════════════════════════
// HITL Feedback & Additional Interactive Pillars
// ═══════════════════════════════════════════════════════════════

let currentQueryId = null;
let currentQueryText = "";
let currentAnswerText = "";

function setupHITLFeedback(queryId, queryText, answerText) {
    currentQueryId = queryId;
    currentQueryText = queryText;
    currentAnswerText = answerText;

    const hitlStatus = $('hitl-status');
    if (hitlStatus) hitlStatus.textContent = "";

    const btnUp = $('hitl-up-btn');
    const btnDown = $('hitl-down-btn');

    if (btnUp) btnUp.onclick = () => submitHITL('up');
    if (btnDown) btnDown.onclick = () => submitHITL('down');
}

async function submitHITL(rating) {
    if (!currentQueryId) return;
    const hitlStatus = $('hitl-status');
    if (hitlStatus) hitlStatus.textContent = "Saving...";

    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query_id: currentQueryId,
                rating: rating,
                query_text: currentQueryText,
                answer_text: currentAnswerText,
            }),
        });

        const res = await response.json();
        if (hitlStatus) {
            hitlStatus.textContent = rating === 'up' ? '✅ Thank you! Flagged as helpful.' : '⚠️ Thank you! Flagged for review.';
        }
        trackTelemetry('hitl_feedback', { rating, query_id: currentQueryId });
    } catch (err) {
        console.error('HITL feedback error:', err);
        if (hitlStatus) hitlStatus.textContent = "Failed to send rating.";
    }
}

// Sticky Mobile Mic CTA
const mobileMicBtn = $('mobile-mic-btn');
if (mobileMicBtn) {
    mobileMicBtn.addEventListener('click', () => {
        const micBtn = els.micButton;
        if (micBtn) micBtn.click();
        els.micButton.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
}

// Privacy & Safety Policy Modal
const privacyLink = $('privacy-link');
const headerPrivacyLink = $('header-privacy-link');
const privacyModal = $('privacy-modal');
const closePrivacyBtn = $('close-privacy-btn');
const closePrivacyBtnBottom = $('close-privacy-btn-bottom');

function openPrivacyModal(e) {
    if (e) e.preventDefault();
    if (privacyModal) privacyModal.showModal();
}

function closePrivacyModal() {
    if (privacyModal) privacyModal.close();
}

if (privacyLink) privacyLink.addEventListener('click', openPrivacyModal);
if (headerPrivacyLink) headerPrivacyLink.addEventListener('click', openPrivacyModal);
if (closePrivacyBtn) closePrivacyBtn.addEventListener('click', closePrivacyModal);
if (closePrivacyBtnBottom) closePrivacyBtnBottom.addEventListener('click', closePrivacyModal);

// Close on clicking backdrop
if (privacyModal) {
    privacyModal.addEventListener('click', (e) => {
        const rect = privacyModal.getBoundingClientRect();
        const isInDialog = (
            rect.top <= e.clientY && e.clientY <= rect.top + rect.height &&
            rect.left <= e.clientX && e.clientX <= rect.left + rect.width
        );
        if (!isInDialog) {
            privacyModal.close();
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// Dynamic Knowledge Ingestion & Teach AI
// ═══════════════════════════════════════════════════════════════

if (els.toggleTeachBtn) {
    els.toggleTeachBtn.addEventListener('click', () => {
        const isHidden = !els.teachDrawer || els.teachDrawer.style.display === 'none';
        if (els.teachDrawer) {
            els.teachDrawer.style.display = isHidden ? 'block' : 'none';
            if (isHidden && els.teachInput) {
                els.teachInput.focus();
                els.teachDrawer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    });
}

if (els.teachDrawerClose) {
    els.teachDrawerClose.addEventListener('click', () => {
        if (els.teachDrawer) els.teachDrawer.style.display = 'none';
    });
}

if (els.teachForm) {
    els.teachForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const factText = els.teachInput ? els.teachInput.value.trim() : '';
        if (!factText) return;
        await submitTeachKnowledge(factText);
    });
}

if (els.resetThreadBtn) {
    els.resetThreadBtn.addEventListener('click', () => {
        state.conversationHistory = [];
        state.sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
        if (els.chatThreadWrapper) els.chatThreadWrapper.style.display = 'none';
        if (els.chatThreadMessages) els.chatThreadMessages.innerHTML = '';
        showStatus('✨ Started fresh conversation session.');
        setTimeout(hideStatus, 1500);
    });
}

async function submitTeachKnowledge(factText) {
    if (els.teachSubmitBtn) els.teachSubmitBtn.disabled = true;
    if (els.teachStatus) els.teachStatus.innerHTML = '<span class="status-spinner-inline"></span> Expanding knowledge with Gemini & indexing vectors...';
    showStatus('Expanding knowledge with Gemini & generating FAISS embeddings...');

    try {
        const res = await fetch('/api/learn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fact: factText,
                language: state.selectedLanguage === 'auto' ? 'eng_Latn' : state.selectedLanguage,
            }),
        });

        const data = await res.json();
        if (res.ok && data.success) {
            if (els.teachStatus) {
                els.teachStatus.innerHTML = `<span style="color: #22c55e; font-weight: 700;">✅ Ingested ${data.passages_added} passages! (Total: ${data.total_vectors} vectors)</span>`;
            }
            if (els.vectorsBadge) {
                els.vectorsBadge.textContent = `${data.total_vectors} vectors`;
                els.vectorsBadge.classList.add('badge-highlight');
                setTimeout(() => els.vectorsBadge.classList.remove('badge-highlight'), 3000);
            }
            if (els.teachInput) els.teachInput.value = '';

            // Add note to conversation history so user can query it
            state.conversationHistory.push({
                role: 'assistant',
                text: `🧠 **Knowledge Ingested:** ${factText}\n\nIndexed ${data.passages_added} dynamic passages into vector store. Ask me anything about this topic!`,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            });
            renderChatThread();

            setTimeout(() => {
                if (els.teachDrawer) els.teachDrawer.style.display = 'none';
                if (els.teachStatus) els.teachStatus.textContent = '';
                if (els.teachSubmitBtn) els.teachSubmitBtn.disabled = false;
            }, 2200);
        } else {
            if (els.teachStatus) {
                els.teachStatus.innerHTML = `<span style="color: #ef4444;">❌ Error: ${data.detail || 'Failed to learn'}</span>`;
            }
            if (els.teachSubmitBtn) els.teachSubmitBtn.disabled = false;
        }
    } catch (err) {
        console.error('Teach error:', err);
        if (els.teachStatus) els.teachStatus.innerHTML = '<span style="color: #ef4444;">❌ Network error</span>';
        if (els.teachSubmitBtn) els.teachSubmitBtn.disabled = false;
    } finally {
        hideStatus();
    }
}

// Initial checks
checkHealth();
fetchAnalytics();

// Poll health every 30s
setInterval(checkHealth, 30000);

